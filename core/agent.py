"""Production-style GOFO agent wrapper for CLI, API, and dashboard clients."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import config
from core.clarification_manager import (
    ClarificationManager,
    PendingClarification,
    build_clarification_response,
)
from core.error_handler import handle_error
from core.intent_classifier import IntentClassifier, IntentClassification, IntentType
from core.intent_router import IntentRouter, RouteIntent
from core.logger import get_logger
from core.models import QueryResponse
from core.plan_executor import PlanExecutor
from core.planner import Planner
from core.quality_assurance import (
    DecisionEngine,
    QAAction,
    QualityAssurancePipeline,
    QualityReport,
    build_answer_context_from_response,
)
from core.reflection import ReflectionAgent, ReflectionResult
from core.route_dispatcher import DispatchContext, RouteDispatcher
from tools.context import build_context
from tools.conversation import ConversationResolver
from tools.files.attachment_memory import AttachmentMemory
from tools.files.service import AttachmentService
from tools.memory import (
    ConversationMemory,
    ConversationState,
    references_previous_result,
    resolve,
)
from tools.orchestration import analyze_request
from tools.planner.intent_classifier import classify_intent as classify_business_intent

logger = get_logger("agent")


class GOFOAgent:
    """Stateful wrapper around the existing GOFO agent core."""

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.state = ConversationState()
        self.conversation = ConversationResolver()
        self.attachment_memory = AttachmentMemory()
        self.attachment_service = AttachmentService()
        self.intent_router = IntentRouter()
        self.route_dispatcher = RouteDispatcher()
        self.intent_classifier = IntentClassifier()
        self.planner = Planner()
        self.plan_executor = PlanExecutor()
        self.quality_assurance = QualityAssurancePipeline(
            decision_engine=DecisionEngine(
                score_threshold=config.QA_SCORE_THRESHOLD,
                approve_threshold=config.QA_APPROVE_THRESHOLD,
                max_retries=config.QA_MAX_RETRIES,
            )
        )
        self.reflection_agent = ReflectionAgent(
            qa_pipeline=self.quality_assurance,
            use_llm=config.REFLECTION_USE_LLM,
            max_retries=config.REFLECTION_MAX_RETRIES,
            approve_confidence=config.QA_APPROVE_THRESHOLD,
        )
        self.clarification_manager = ClarificationManager()
        self.pending_clarification: PendingClarification | None = None

    def ask(self, question: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        """Ask GOFO a question and return a dashboard-friendly response."""
        started = perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        logger.info("Incoming question")
        attachment_ids = attachment_ids or []
        try:
            if attachment_ids:
                processed = self.attachment_service.get_many_processed(attachment_ids, question=question)
                self.attachment_memory.register_contexts(processed)

            previous_state = self.state.snapshot()
            previous_semantic_state = self.conversation.snapshot()

            # Stage 1: Intent classification (primary "what" decision).
            classification: IntentClassification | None = None
            if config.INTENT_CLASSIFIER_ENABLED:
                classification = self.intent_classifier.classify(question, self.memory)

            # Attachment activate/detach + wait-for-upload remain router-owned.
            route_decision = self.intent_router.route(
                question,
                previous_state,
                new_attachment_ids=attachment_ids or None,
                has_stored_attachments=bool(self.attachment_memory.processed_contexts),
            )
            if classification is not None:
                classification = _align_classification_with_route(
                    classification,
                    route_decision,
                    has_new_upload=bool(attachment_ids),
                )

            if route_decision["detach_attachments"]:
                self.attachment_memory.deactivate()
                self.state.attachment_active = False
            elif route_decision["use_attachments"]:
                self.attachment_memory.activate(attachment_ids or None)

            attachment_contexts, file_context = self.attachment_memory.resolve_for_question(
                question,
                attachment_ids=attachment_ids or None,
                use_attachments=route_decision["use_attachments"],
            )

            conversation_resolution = self.conversation.resolve(question)
            semantic = analyze_request(
                question,
                resolved_question=conversation_resolution.resolved_question,
                conversation_state=previous_semantic_state,
                repair_detected=conversation_resolution.repair_detected,
                has_attachments=route_decision["use_attachments"] and bool(attachment_contexts),
            )
            intent = classify_business_intent(
                question,
                resolved_question=conversation_resolution.resolved_question,
                conversation_state=previous_semantic_state,
                repair_detected=conversation_resolution.repair_detected,
                intent_hint=_routing_intent_hint(route_decision, question)
                or conversation_resolution.intent_hint
                or semantic.intent_hint(),
            )
            context = build_context(conversation_resolution, intent, previous_semantic_state)

            resolved_question = context.resolved_question
            if semantic.confidence >= 0.65 and semantic.resolved_question:
                resolved_question = semantic.resolved_question
            if (
                resolved_question == question
                and self.memory.get_recent_history()
                and not conversation_resolution.repair_detected
                and not conversation_resolution.intent_hint
                and not route_decision["use_attachments"]
            ):
                resolved_question = resolve(question, self.memory)

            # Resume paused plan when the user answers a pending clarification.
            resumed_from_clarification = False
            clarification_slots: dict[str, str] = {}
            if (
                config.CLARIFICATION_MANAGER_ENABLED
                and self.pending_clarification is not None
                and not attachment_ids
            ):
                resume_decision = self.clarification_manager.apply_user_response(
                    self.pending_clarification,
                    question,
                )
                if resume_decision.resumed_question:
                    resolved_question = resume_decision.resumed_question
                    resumed_from_clarification = True
                    clarification_slots = dict(resume_decision.filled_slots or {})
                    # Persist filled slots into short-term memory for follow-ups.
                    if clarification_slots.get("metric"):
                        self.memory.global_state["current_metric"] = clarification_slots["metric"]
                    if clarification_slots.get("date_range"):
                        self.memory.global_state["date_range"] = clarification_slots["date_range"]
                    self.pending_clarification = None
                    logger.info("Resumed after clarification: %s", resolved_question)

            logger.info("Resolved question: %s", resolved_question)
            logger.info(
                "Semantic analysis: domain=%s sub_intent=%s capability=%s confidence=%s",
                semantic.domain,
                semantic.sub_intent,
                semantic.capability,
                semantic.confidence,
            )

            result_context = None
            if (
                context.use_previous_result
                or references_previous_result(question)
                or references_previous_result(resolved_question)
                or route_decision["followup"]
                or (classification is not None and classification.requires_memory)
            ):
                result_context = self.memory.get_current_state().get("last_result_context")

            cached = self.conversation.cached_response(conversation_resolution)
            sql_cache_hit = cached is not None
            last_analysis = self.attachment_memory.last_analysis or {}

            # Multi-step planner execution for analytical / dashboard / SOP-compare.
            # Attachments, wait-for-upload, and cache hits stay on RouteDispatcher.
            planner_intents = {
                IntentType.SQL_Query,
                IntentType.SQL_Analysis,
                IntentType.Dashboard,
                IntentType.SOP_QA,
                IntentType.SOP_Summary,
                IntentType.SOP_Compare,
                IntentType.Follow_Up,
                IntentType.Explain_Result,
                IntentType.Unknown,
            }
            use_planner_pipeline = (
                config.PLANNER_ENABLED
                and config.INTENT_CLASSIFIER_ENABLED
                and classification is not None
                and (
                    classification.intent in planner_intents
                    or resumed_from_clarification
                )
                and route_decision["intent"]
                not in {
                    RouteIntent.WAIT_FOR_UPLOAD,
                    RouteIntent.ATTACHMENT_ANALYSIS,
                    RouteIntent.ATTACHMENT_VISUALIZATION,
                }
                and cached is None
            )

            if use_planner_pipeline:
                plan = self.planner.plan(
                    resolved_question,
                    classification,
                    self.memory,
                    attachment_context={
                        "attachment_ids": attachment_ids,
                        "filenames": [ctx.filename for ctx in attachment_contexts],
                        "use_attachments": route_decision["use_attachments"],
                    },
                )

                # Clarification Manager: ask before Tool Orchestration when needed.
                if config.CLARIFICATION_MANAGER_ENABLED and not resumed_from_clarification:
                    clarify = self.clarification_manager.evaluate(
                        question,
                        resolved_question=resolved_question,
                        classification=classification,
                        plan=plan,
                        conversation_memory=self.memory,
                        attachment_context={
                            "attachment_ids": attachment_ids,
                            "filenames": [ctx.filename for ctx in attachment_contexts],
                            "use_attachments": route_decision["use_attachments"],
                        },
                    )
                    if clarify.needs_clarification and clarify.pending is not None:
                        self.pending_clarification = clarify.pending
                        response = build_clarification_response(
                            clarify,
                            question=question,
                            classification=classification,
                        )
                        response.execution_plan = plan.model_dump()
                        response.resolved_question = resolved_question
                        # Skip tool execution / QA; fall through to memory update.
                        quality_report = None
                        reflection_result = None
                        qa_retry_count = 0
                        reflection_retry_count = 0
                        # Jump to shared response finalization by using a flag.
                        skip_tools = True
                    else:
                        skip_tools = False
                else:
                    skip_tools = False

                if not skip_tools:
                    execution = self.plan_executor.execute(
                        plan,
                        question=question,
                        resolved_question=resolved_question,
                        conversation_memory=self.memory,
                        attachment_contexts=attachment_contexts,
                        file_context=file_context,
                        attachment_ids=attachment_ids,
                        stored_attachments=self.attachment_memory.get_active_stored()
                        or list(self.attachment_memory.stored_attachments.values()),
                        previous_filter=self.attachment_memory.active_filters or None,
                        last_entity=(last_analysis.get("last_entity") or None),
                        result_context=result_context,
                    )
                    response = execution.response
                    response.intent_classification = classification.model_dump()
                    response.execution_plan = plan.model_dump()
                    response.step_results_summary = execution.response.step_results_summary
                    if response.capability == "sql" and response.sql_rows:
                        from tools.analyzer.business_reasoner import analyze_business_response

                        response = analyze_business_response(response, resolved_question)
                        response.intent_classification = classification.model_dump()
                        response.execution_plan = plan.model_dump()
            else:
                skip_tools = False
                response = self.route_dispatcher.dispatch(
                    DispatchContext(
                        question=question,
                        resolved_question=resolved_question,
                        route=route_decision,
                        attachment_contexts=attachment_contexts,
                        file_context=file_context,
                        attachment_ids=attachment_ids,
                        intent=intent,
                        conversation_state=previous_semantic_state,
                        semantic=semantic,
                        result_context=result_context,
                        cached_response=cached,
                        stored_attachments=self.attachment_memory.get_active_stored()
                        or list(self.attachment_memory.stored_attachments.values()),
                        previous_filter=self.attachment_memory.active_filters or None,
                        last_entity=(last_analysis.get("last_entity") or None),
                    )
                )
                if response is None:
                    raise RuntimeError("Route dispatcher returned no response.")
                if classification is not None:
                    response.intent_classification = classification.model_dump()

            if not skip_tools:
                quality_report: QualityReport | None = None
                reflection_result: ReflectionResult | None = None
                qa_retry_count = 0
                reflection_retry_count = 0

                # Reflection is the primary critic. It uses QA validators underneath and
                # sends structured feedback to the Planner (never executes tools itself).
                if config.REFLECTION_ENABLED and _should_run_qa(response, route_decision):
                    reflection_result, response, reflection_retry_count = self._run_reflection(
                        question=question,
                        resolved_question=resolved_question,
                        response=response,
                        classification=classification,
                        attachment_contexts=attachment_contexts,
                        file_context=file_context,
                        attachment_ids=attachment_ids,
                        result_context=result_context,
                        last_analysis=last_analysis,
                    )
                    response.reflection_result = reflection_result.model_dump()
                    response.reflection_retry_count = reflection_retry_count
                    if reflection_result.quality_report:
                        response.quality_report = reflection_result.quality_report
                        response.qa_retry_count = reflection_retry_count
                    if reflection_result.should_ask_user:
                        response.answer = (
                            reflection_result.clarification_question
                            or _qa_clarification_message_from_reflection(reflection_result)
                        )
                        response.capability = "clarification"
                        # Store pending from reflection ask-user so the next turn can resume.
                        if config.CLARIFICATION_MANAGER_ENABLED and self.pending_clarification is None:
                            self.pending_clarification = PendingClarification(
                                original_question=question,
                                missing_fields=["date_range"],
                                pending_question=response.answer or "",
                                options=[],
                                reason="Reflection requested user clarification.",
                                ambiguity_type="reflection",
                                classification=(
                                    classification.model_dump() if classification else None
                                ),
                                draft_plan=response.execution_plan,
                                resolved_base=resolved_question,
                            )
                elif config.QUALITY_ASSURANCE_ENABLED and _should_run_qa(response, route_decision):
                    quality_report, response, qa_retry_count = self._run_quality_assurance(
                        question=question,
                        resolved_question=resolved_question,
                        response=response,
                        classification=classification,
                        attachment_contexts=attachment_contexts,
                        file_context=file_context,
                        attachment_ids=attachment_ids,
                        result_context=result_context,
                        last_analysis=last_analysis,
                    )
                    response.quality_report = quality_report.model_dump()
                    response.qa_retry_count = qa_retry_count
                    if quality_report.should_ask_user:
                        response.answer = _qa_clarification_message(quality_report)
                        response.capability = "clarification"

            logger.info("Selected capability: %s", response.capability)
            if response.generated_sql:
                logger.info("Generated SQL: %s", response.generated_sql)

            # Keep clarification-filled slots on the response so memory.add_turn
            # does not overwrite them with heuristic inference.
            if clarification_slots.get("metric"):
                response.business_metric = clarification_slots["metric"]
                entities = dict(response.planning_entities or {})
                entities["metric"] = clarification_slots["metric"]
                response.planning_entities = entities
            if clarification_slots.get("date_range"):
                response.date_range = clarification_slots["date_range"]

            self.memory.add_turn(
                user_question=question,
                resolved_question=resolved_question,
                response=response,
            )
            self.state.update(
                user_question=question,
                resolved_question=resolved_question,
                response=response,
                route_decision=route_decision,
            )
            new_state = self.state.snapshot()
            new_semantic_state = self.conversation.update(
                original_question=question,
                resolved_question=resolved_question,
                response=response,
                intent=(
                    classification.intent.value
                    if classification is not None
                    else intent.value
                ),
                inherited_context=context.inherited_context,
                cache_key=context.cache_key,
            )
            logger.info("Memory updated")
            logger.info("Previous state: %s", previous_state)
            logger.info("New state: %s", new_state)
            response.original_question = question
            response.resolved_question = resolved_question
            response.repair_detected = conversation_resolution.repair_detected
            response.repair_type = conversation_resolution.repair_type
            response.changed_dimension = conversation_resolution.changed_dimension
            # Keep business Intent (ROOT_CAUSE, etc.) on classifier_intent for compatibility.
            # The new IntentType taxonomy lives in intent_classification.
            response.classifier_intent = intent.value
            response.inherited_context = context.inherited_context
            response.semantic_domain = semantic.domain
            response.semantic_sub_intent = semantic.sub_intent
            response.response_mode = semantic.response_mode
            response.semantic_reasoning = semantic.reasoning
            response.sql_cache_hit = sql_cache_hit
            response.memory_updated = True
            response.attachment_ids = [context.attachment_id for context in attachment_contexts] or []
            response.attachment_filenames = [context.filename for context in attachment_contexts] or []
            response.data_sources = route_decision.get("data_sources") or []
            analysis_summary = dict(response.file_context_summary or {})
            response.file_context_summary = {
                **_compact_file_context(file_context),
                **{key: value for key, value in analysis_summary.items() if value is not None},
            }
            self.attachment_memory.update_analysis(
                {
                    "file_context": file_context,
                    "data_sources": response.data_sources,
                    "answer": response.answer,
                    "route": route_decision,
                    "active_filter": (response.analysis_filters or {})
                    or analysis_summary.get("active_filter")
                    or {},
                    "last_entity": analysis_summary.get("last_entity")
                    or last_analysis.get("last_entity"),
                    "charts": response.charts or [],
                    "intent_classification": response.intent_classification,
                    "execution_plan": response.execution_plan,
                }
            )
            response.memory_history_count = len(self.memory.get_recent_history())
            response.memory_current_state = self.memory.get_current_state()
            response.memory_current_state["conversation_state"] = new_state
            response.memory_current_state["previous_conversation_state"] = previous_state
            response.memory_current_state["semantic_conversation"] = new_semantic_state
            response.memory_current_state["previous_semantic_conversation"] = previous_semantic_state
            response.memory_current_state["attachment_memory"] = self.attachment_memory.snapshot()
            response.memory_current_state["route_decision"] = route_decision
            if classification is not None:
                response.memory_current_state["intent_classification"] = classification.model_dump()
            if response.execution_plan:
                response.memory_current_state["execution_plan"] = response.execution_plan
            if response.quality_report:
                response.memory_current_state["quality_report"] = response.quality_report
            if response.reflection_result:
                response.memory_current_state["reflection_result"] = response.reflection_result
            if self.pending_clarification is not None:
                response.memory_current_state["pending_clarification"] = (
                    self.pending_clarification.model_dump()
                )
            response.last_result_context = self.memory.get_current_state().get("last_result_context")
            return format_agent_response(response)
        except Exception as exc:
            logger.exception("Agent error: %s", handle_error(exc))
            raise
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info("Execution time: %sms", elapsed_ms)

    def _run_reflection(
        self,
        *,
        question: str,
        resolved_question: str,
        response: QueryResponse,
        classification: IntentClassification | None,
        attachment_contexts: list[Any],
        file_context: dict[str, Any],
        attachment_ids: list[str],
        result_context: dict[str, Any] | None,
        last_analysis: dict[str, Any],
    ) -> tuple[ReflectionResult, QueryResponse, int]:
        """Critique the answer and optionally retry via Planner (max REFLECTION_MAX_RETRIES)."""
        retry_count = 0
        current = response
        previous_answer = current.answer
        original_capability = (response.capability or "").lower()
        from core.quality_assurance import is_file_capability

        file_answer = is_file_capability(original_capability)

        reflection = self.reflection_agent.critique_response(
            resolved_question or question,
            current,
            conversation_memory=self.memory,
            ambiguous=_looks_ambiguous(resolved_question or question, classification),
            retry_count=retry_count,
            previous_answer=None,
        )

        # Attachment/ADA answers must not be overwritten by warehouse SQL retries.
        if file_answer and (
            reflection.should_retry_sql
            or "SQL" in (reflection.suggested_tool_calls or [])
        ):
            reflection = reflection.model_copy(
                update={
                    "approved": True,
                    "should_retry_sql": False,
                    "should_retry_retrieval": False,
                    "should_retry_python": False,
                    "suggested_tool_calls": [
                        tool
                        for tool in (reflection.suggested_tool_calls or [])
                        if tool.upper() not in {"SQL", "RAG"}
                    ],
                    "feedback": list(
                        dict.fromkeys(
                            (reflection.feedback or [])
                            + ["Kept attachment analysis; skipped warehouse SQL retry."]
                        )
                    ),
                    "reasoning": (reflection.reasoning or "")
                    + " | attachment capability — SQL retry suppressed",
                }
            )

        while (
            not reflection.approved
            and not reflection.should_ask_user
            and retry_count < config.REFLECTION_MAX_RETRIES
            and reflection.needs_retry()
        ):
            if file_answer and reflection.should_retry_sql and not reflection.should_retry_python:
                # Do not replace a good file answer with SELECT UNKNOWN from ops DB.
                reflection = reflection.model_copy(
                    update={
                        "approved": True,
                        "should_retry_sql": False,
                        "suggested_tool_calls": [],
                    }
                )
                break
            retry_count += 1
            logger.info(
                "Reflection retry %s/%s tools=%s missing=%s",
                retry_count,
                config.REFLECTION_MAX_RETRIES,
                reflection.suggested_tool_calls,
                reflection.missing_information,
            )
            if classification is None:
                break

            retry_plan = self.planner.plan_retry(
                resolved_question or question,
                classification,
                reflection,
                previous_plan=current.execution_plan,
                previous_answer=previous_answer,
                conversation_memory=self.memory,
            )
            if config.DEBUG:
                logger.info("Planner decision after reflection: %s", retry_plan.goal)

            if retry_plan.requires_clarification:
                reflection = reflection.model_copy(
                    update={
                        "should_ask_user": True,
                        "approved": False,
                        "clarification_question": retry_plan.clarification_question
                        or reflection.clarification_question
                        or _default_reflection_clarification(reflection),
                    }
                )
                break

            execution = self.plan_executor.execute(
                retry_plan,
                question=question,
                resolved_question=resolved_question,
                conversation_memory=self.memory,
                attachment_contexts=attachment_contexts,
                file_context=file_context,
                attachment_ids=attachment_ids,
                stored_attachments=self.attachment_memory.get_active_stored()
                or list(self.attachment_memory.stored_attachments.values()),
                previous_filter=self.attachment_memory.active_filters or None,
                last_entity=(last_analysis.get("last_entity") or None),
                result_context=result_context,
            )
            retried = execution.response
            if file_answer and (
                "UNKNOWN" in (retried.generated_sql or "").upper()
                or (retried.answer or "") == "No matching operational records were found."
            ):
                logger.info("Preserving attachment answer; discarding SQL UNKNOWN retry")
                reflection = reflection.model_copy(
                    update={
                        "approved": True,
                        "should_retry_sql": False,
                        "suggested_tool_calls": [],
                        "feedback": list(
                            dict.fromkeys(
                                (reflection.feedback or [])
                                + ["Preserved attachment analysis over failed SQL retry."]
                            )
                        ),
                    }
                )
                break

            current = retried
            current.intent_classification = classification.model_dump()
            current.execution_plan = retry_plan.model_dump()
            current.step_results_summary = execution.response.step_results_summary
            if file_answer and not current.attachment_filenames and response.attachment_filenames:
                current.attachment_filenames = list(response.attachment_filenames)
            if file_answer and response.file_context_summary and not current.file_context_summary:
                current.file_context_summary = response.file_context_summary
            if current.capability == "sql" and current.sql_rows:
                from tools.analyzer.business_reasoner import analyze_business_response

                current = analyze_business_response(current, resolved_question)
                current.intent_classification = classification.model_dump()
                current.execution_plan = retry_plan.model_dump()

            current.plan_reason = (
                f"Reflection retry {retry_count}: "
                f"{', '.join(reflection.feedback[:2]) or 'improve evidence'}"
            )
            previous_answer = previous_answer or response.answer
            reflection = self.reflection_agent.critique_response(
                resolved_question or question,
                current,
                conversation_memory=self.memory,
                ambiguous=False,
                retry_count=retry_count,
                previous_answer=previous_answer,
            )

        if config.DEBUG:
            logger.info(
                "Reflection final approved=%s confidence=%s retries=%s",
                reflection.approved,
                reflection.confidence,
                retry_count,
            )
        return reflection, current, retry_count

    def _run_quality_assurance(
        self,
        *,
        question: str,
        resolved_question: str,
        response: QueryResponse,
        classification: IntentClassification | None,
        attachment_contexts: list[Any],
        file_context: dict[str, Any],
        attachment_ids: list[str],
        result_context: dict[str, Any] | None,
        last_analysis: dict[str, Any],
        route_decision: dict[str, Any],
    ) -> tuple[QualityReport, QueryResponse, int]:
        """Evaluate the answer and optionally retry via Planner (max QA_MAX_RETRIES)."""
        del route_decision  # reserved for future route-aware QA policies
        retry_count = 0
        current = response
        report = self.quality_assurance.evaluate(
            build_answer_context_from_response(
                question=resolved_question or question,
                response=current,
                conversation_memory=self.memory,
                ambiguous=_looks_ambiguous(resolved_question or question, classification),
            ),
            retry_count=retry_count,
        )

        while (
            not report.approved
            and not report.should_ask_user
            and retry_count < config.QA_MAX_RETRIES
            and report.action
            in {
                QAAction.RETRY_RETRIEVAL,
                QAAction.RETRY_SQL,
                QAAction.RETRY_PYTHON,
                QAAction.RETRY_PLAN,
            }
        ):
            retry_count += 1
            logger.info(
                "QA retry %s/%s action=%s reason=%s",
                retry_count,
                config.QA_MAX_RETRIES,
                report.action.value,
                report.retry_reason,
            )
            if classification is None:
                break

            retry_plan = self.planner.plan_retry(
                resolved_question or question,
                classification,
                report,
                previous_plan=current.execution_plan,
                previous_answer=current.answer,
                conversation_memory=self.memory,
            )
            if retry_plan.requires_clarification:
                report = report.model_copy(
                    update={
                        "should_ask_user": True,
                        "approved": False,
                        "action": QAAction.ASK_USER,
                        "retry_reason": report.retry_reason or "Clarification required after QA",
                    }
                )
                break

            execution = self.plan_executor.execute(
                retry_plan,
                question=question,
                resolved_question=resolved_question,
                conversation_memory=self.memory,
                attachment_contexts=attachment_contexts,
                file_context=file_context,
                attachment_ids=attachment_ids,
                stored_attachments=self.attachment_memory.get_active_stored()
                or list(self.attachment_memory.stored_attachments.values()),
                previous_filter=self.attachment_memory.active_filters or None,
                last_entity=(last_analysis.get("last_entity") or None),
                result_context=result_context,
            )
            current = execution.response
            current.intent_classification = classification.model_dump()
            current.execution_plan = retry_plan.model_dump()
            current.step_results_summary = execution.response.step_results_summary
            if current.capability == "sql" and current.sql_rows:
                from tools.analyzer.business_reasoner import analyze_business_response

                current = analyze_business_response(current, resolved_question)
                current.intent_classification = classification.model_dump()
                current.execution_plan = retry_plan.model_dump()

            # Preserve prior answer + QA feedback for iterative improvement traces.
            current.plan_reason = (
                f"QA retry {retry_count}: {report.retry_reason or report.action.value}"
            )
            report = self.quality_assurance.evaluate(
                build_answer_context_from_response(
                    question=resolved_question or question,
                    response=current,
                    conversation_memory=self.memory,
                    ambiguous=False,
                ),
                retry_count=retry_count,
            )

        if config.DEBUG:
            logger.info(
                "QA final decision approved=%s action=%s retries=%s",
                report.approved,
                report.action.value,
                retry_count,
            )
        return report, current, retry_count


def format_agent_response(response: QueryResponse) -> dict[str, Any]:
    """Convert QueryResponse into the API/dashboard contract."""
    memory_state = response.memory_current_state or {}
    route_decision = memory_state.get("route_decision") or {}
    return {
        "answer": response.answer or "",
        "sources": [source.model_dump() for source in response.sources],
        "sql": response.generated_sql or "",
        "data": response.sql_rows or [],
        "analysis": {
            "capability": response.capability,
            "root_cause": response.root_cause or {},
            "anomaly": response.anomaly or {},
            "long_memory_matches": response.long_memory_matches or [],
            "previous_issues_found": response.previous_issues_found or [],
            "metric": response.business_metric,
            "dimension": response.analysis_dimension,
            "date_range": response.date_range,
            "filters": response.analysis_filters or {},
            "intent": response.classifier_intent or response.planning_intent,
            "primary_intent": (response.intent_classification or {}).get("intent"),
            "intent_classification": response.intent_classification or {},
            "execution_plan": response.execution_plan or {},
            "step_results_summary": response.step_results_summary or [],
            "inherited_context": response.inherited_context or {},
            "business_findings": response.business_findings or [],
            "memory_updated": bool(response.memory_updated),
            "sql_cache_hit": bool(response.sql_cache_hit),
            "repair_detected": response.repair_detected,
            "resolved_question": response.resolved_question or response.question,
            "memory_turns": response.memory_history_count or 0,
            "previous_question": memory_state.get("previous_question"),
            "previous_answer": memory_state.get("previous_answer"),
            "current_entities": memory_state.get("active_entities", {}),
            "previous_state": memory_state.get("previous_conversation_state", {}),
            "new_state": memory_state.get("conversation_state", {}),
            "semantic_state": memory_state.get("semantic_conversation", {}),
            "attachment_ids": response.attachment_ids or [],
            "attachment_filenames": response.attachment_filenames or [],
            "data_sources": response.data_sources or [],
            "file_context_summary": response.file_context_summary or {},
            "attachment_memory": memory_state.get("attachment_memory", {}),
            "semantic_domain": response.semantic_domain,
            "semantic_sub_intent": response.semantic_sub_intent,
            "response_mode": response.response_mode,
            "semantic_reasoning": response.semantic_reasoning,
            "suggested_next_steps": response.suggested_next_steps or [],
            "route_intent": route_decision.get("intent"),
            "route_confidence": route_decision.get("confidence"),
            "route_followup": route_decision.get("followup"),
            "route_handler": route_decision.get("handler"),
            "attachment_active": route_decision.get("attachment_active"),
            "chart_type": route_decision.get("chart_type"),
            "charts": response.charts or [],
            "quality_report": response.quality_report or {},
            "qa_retry_count": response.qa_retry_count or 0,
            "reflection_result": response.reflection_result or {},
            "reflection_retry_count": response.reflection_retry_count or 0,
            "agent_state": response.agent_state or {},
            "requires_clarification": bool(response.requires_clarification)
            or response.capability == "clarification",
            "clarification_question": response.clarification_question,
            "clarification_options": response.clarification_options or [],
            "missing_fields": response.missing_fields or [],
        },
        "charts": response.charts or [],
        "recommendations": _merge_recommendations(response),
        "kpi": _kpi_payload(response),
        "raw": response.model_dump(),
    }


def _recommendations(recommendation: str | None) -> list[str]:
    if not recommendation:
        return []
    parts = [part.strip(" -") for part in recommendation.split(";")]
    return [part for part in parts if part]


def _merge_recommendations(response: QueryResponse) -> list[str]:
    recommendations = _recommendations(response.recommendation)
    next_steps = list(response.suggested_next_steps or [])
    merged: list[str] = []
    for item in recommendations + next_steps:
        if item and item not in merged:
            merged.append(item)
    return merged


def _kpi_payload(response: QueryResponse) -> dict[str, Any]:
    kpi = dict(response.kpi_summary or {})
    rows = response.sql_rows or []
    if rows:
        first = rows[0]
        for source_key, target_key in (
            ("pickup_count", "pickup_count"),
            ("completion_rate", "completion_rate"),
            ("failed_pickups", "failed"),
            ("delayed_pickups", "delayed"),
            ("package_volume", "package_volume"),
        ):
            if source_key in first:
                kpi[target_key] = first[source_key]
    return kpi


def _routing_intent_hint(route_decision: dict[str, Any], question: str) -> str | None:
    intent_name = route_decision.get("intent")
    normalized = question.lower()
    if intent_name == "ATTACHMENT_VISUALIZATION":
        return "FILE_VISUALIZATION"
    if intent_name not in {"ATTACHMENT_ANALYSIS", "ATTACHMENT_VISUALIZATION"}:
        return None
    if any(phrase in normalized for phrase in ("summarize", "summary", "what is in")):
        return "FILE_SUMMARY"
    if "compare" in normalized:
        return "FILE_COMPARISON"
    if any(phrase in normalized for phrase in ("visualize", "plot", "chart", "graph")):
        return "FILE_VISUALIZATION"
    if any(phrase in normalized for phrase in ("what should operations do", "recommend")):
        return "FILE_RECOMMENDATION"
    return "FILE_ANALYSIS"


def _should_run_qa(response: QueryResponse, route_decision: dict[str, Any]) -> bool:
    """Skip QA for chat, wait-for-upload, and attachment/ADA answers.

    Attachment analyses store dataframe previews in sql_rows. Running the SQL
    QA/reflection loop overwrites good file summaries with SELECT UNKNOWN.
    """
    intent = route_decision.get("intent")
    if intent in {
        RouteIntent.WAIT_FOR_UPLOAD,
        RouteIntent.GENERAL_CHAT,
        RouteIntent.ATTACHMENT_ANALYSIS,
        RouteIntent.ATTACHMENT_VISUALIZATION,
    }:
        return False
    from core.quality_assurance import is_file_capability

    if is_file_capability(response.capability):
        return False
    if route_decision.get("use_attachments") and not route_decision.get("detach_attachments"):
        # Active attachment session answers are file-grounded.
        if (response.capability or "").lower() in {
            "file_analysis",
            "file_comparison",
            "file_rag_comparison",
            "image_analysis",
            "attachment",
        } or response.attachment_filenames:
            return False
    if (response.capability or "") in {"clarification", "conversation"}:
        return False
    if not (response.answer or "").strip() and not response.sql_rows and not response.sources:
        return False
    return True


def _looks_ambiguous(question: str, classification: IntentClassification | None) -> bool:
    if classification is not None and classification.requires_clarification:
        return True
    normalized = (question or "").lower().strip()
    if not normalized:
        return True
    vague = {
        "help",
        "analyze this",
        "look into it",
        "check performance",
        "what happened",
        "status",
    }
    return normalized in vague


def _qa_clarification_message(report: QualityReport) -> str:
    missing = report.missing_information or []
    missing_block = ""
    if missing:
        missing_block = "\n\nMissing details:\n" + "\n".join(f"• {item}" for item in missing[:6])
    return (
        "I need a bit more detail before I can provide a high-confidence answer.\n\n"
        "Do you want:\n"
        "• today's data\n"
        "• this week's data\n"
        "• all historical data?"
        f"{missing_block}"
    )


def _qa_clarification_message_from_reflection(result: ReflectionResult) -> str:
    return result.clarification_question or _default_reflection_clarification(result)


def _default_reflection_clarification(result: ReflectionResult) -> str:
    missing = result.missing_information or []
    missing_block = ""
    if missing:
        missing_block = "\n\nI still need:\n" + "\n".join(f"• {item}" for item in missing[:6])
    return (
        "Which date range would you like to compare?\n\n"
        "For example:\n"
        "• today vs yesterday\n"
        "• this week vs last week\n"
        "• this month vs last month"
        f"{missing_block}"
    )


def _align_classification_with_route(
    classification: IntentClassification,
    route_decision: dict[str, Any],
    *,
    has_new_upload: bool,
) -> IntentClassification:
    """Keep attachment-session semantics when the router activates attachments."""
    route_intent = route_decision.get("intent")
    if route_intent == RouteIntent.WAIT_FOR_UPLOAD:
        return classification.model_copy(
            update={
                "intent": IntentType.Upload_File,
                "requires_planner": True,
                "requires_memory": True,
                "requires_clarification": False,
                "reasoning": (classification.reasoning or "") + " | aligned to WAIT_FOR_UPLOAD route",
            }
        )
    if route_intent in {RouteIntent.ATTACHMENT_ANALYSIS, RouteIntent.ATTACHMENT_VISUALIZATION} or (
        has_new_upload and route_decision.get("use_attachments")
    ):
        if classification.intent in {
            IntentType.Greeting,
            IntentType.ChitChat,
            IntentType.SQL_Query,
            IntentType.SQL_Analysis,
            IntentType.SOP_QA,
            IntentType.SOP_Summary,
            IntentType.SOP_Compare,
            IntentType.General_Knowledge,
            IntentType.Coding,
        }:
            # Respect explicit SQL/SOP/chat detach decisions from the router.
            if route_decision.get("detach_attachments"):
                return classification
        if classification.intent not in {IntentType.Upload_File, IntentType.Dashboard, IntentType.Follow_Up}:
            return classification.model_copy(
                update={
                    "intent": IntentType.Upload_File,
                    "requires_planner": True,
                    "requires_memory": True,
                    "requires_sql": False,
                    "requires_rag": False,
                    "reasoning": (classification.reasoning or "") + " | aligned to attachment route",
                }
            )
    return classification


def _compact_file_context(file_context: dict[str, Any]) -> dict[str, Any]:
    if not file_context:
        return {}
    return {
        "attachment_ids": file_context.get("attachment_ids", []),
        "filenames": file_context.get("filenames", []),
        "file_types": file_context.get("file_types", []),
        "summaries": file_context.get("summaries", []),
        "statistics": file_context.get("statistics", []),
        "source_references": file_context.get("source_references", []),
    }
