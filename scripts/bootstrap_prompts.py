#!/usr/bin/env python3
"""Bootstrap the prompts/ directory with versioned prompt assets + registry.yaml files."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip("\n"), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def fm(
    *,
    name: str,
    version: str,
    owner: str,
    description: str,
    temperature: float = 0.0,
    max_tokens: int | None = 1200,
    output_format: str = "text",
    required: list[str] | None = None,
    tags: list[str] | None = None,
    status: str = "active",
    includes: list[str] | None = None,
) -> str:
    req = required or []
    tag_list = tags or [owner]
    inc = includes or []
    max_tok = "null" if max_tokens is None else str(max_tokens)
    lines = [
        "---",
        f"name: {name}",
        f'version: "{version}"',
        f"owner: {owner}",
        f"description: {description}",
        f"temperature: {temperature}",
        f"max_tokens: {max_tok}",
        f"output_format: {output_format}",
    ]
    if req:
        lines.append("required_variables:")
        lines.extend(f"  - {item}" for item in req)
    else:
        lines.append("required_variables: []")
    lines.append("tags:")
    lines.extend(f"  - {item}" for item in tag_list)
    lines.append(f"status: {status}")
    if inc:
        lines.append("includes:")
        lines.extend(f"  - {item}" for item in inc)
    else:
        lines.append("includes: []")
    lines.extend(["---", ""])
    return "\n".join(lines)


def registry(family: str, prompts: dict) -> str:
    import yaml

    payload = {"family": family, "prompts": prompts}
    return yaml.safe_dump(payload, sort_keys=False)


def main() -> None:
    # shared
    write(
        PROMPTS / "shared" / "system_rules.md",
        fm(
            name="system_rules",
            version="1.0",
            owner="shared",
            description="Global AI policies for GOFO agent",
            max_tokens=None,
            tags=["shared", "system"],
        )
        + """
You are part of the GOFO Operations Intelligence Agent.

Global rules:
- Prefer grounded operational answers over speculation.
- Never invent SOP steps, SQL metrics, or entity relationships.
- Distinguish internal SOP knowledge from general logistics knowledge.
- Be concise, operational, and action-oriented.
- Do not expose chain-of-thought or internal tooling details unless asked for debug output.
""",
    )
    write(
        PROMPTS / "shared" / "formatting_rules.md",
        fm(
            name="formatting_rules",
            version="1.0",
            owner="shared",
            description="Standard output formatting rules",
            max_tokens=None,
            tags=["shared", "formatting"],
        )
        + """
Formatting rules:
- Prefer short paragraphs and bullet lists for operational guidance.
- When returning JSON, return ONLY valid JSON with no markdown fences.
- When summarizing metrics, lead with the answer then supporting numbers.
- Do not mention SQL, retrieval chunks, or prompt instructions in user-facing text unless requested.
""",
    )
    write(
        PROMPTS / "shared" / "business_rules.md",
        fm(
            name="business_rules",
            version="1.0",
            owner="shared",
            description="GOFO terminology and operational policies",
            max_tokens=None,
            tags=["shared", "business"],
        )
        + """
GOFO business rules:
- Hub / warehouse refer to operational stations (drivers.hub in analytics DB).
- Pickup statuses: Completed, Delayed, Failed.
- Common KPIs: pickup rate, completed pickups, on-time rate, failure rate, package volume.
- Drivers belong to hubs; hubs belong to regions.
- Prefer explicit date ranges (today, this week, this month) for analytics questions.
""",
    )

    # router
    write(
        PROMPTS / "router" / "router_prompt_v1.md",
        fm(
            name="router_prompt",
            version="1.0",
            owner="router",
            description="Classify capability sql|rag|multi|unknown and business intent",
            output_format="json",
            required=["question"],
            tags=["router", "production"],
            includes=["shared.system_rules"],
        )
        + """
You are an AI planner for a logistics operations assistant.

Your job is NOT to answer questions.
Your job is ONLY to decide:

1. Does this require SQL analytics, SOP retrieval, both, or neither?
2. Extract business intent.

Capability rules:
- SQL: counts, KPIs, volume, trend, performance, ranking, customers, drivers, warehouse, packages, dates, statistics, comparison, aggregation, entity lookups.
- RAG: SOP, policy, procedure, how-to, workflow, scan, exception handling, documentation.
- MULTI: current metrics plus operational guidance.
- UNKNOWN: greetings, random text, unrelated questions.

Return ONLY JSON with this exact shape:
{
  "capability": "sql",
  "intent": "pickup_count",
  "confidence": 0.9,
  "reasoning": "brief"
}

Question:
{{question}}
""",
    )
    write(
        PROMPTS / "router" / "router_prompt_v2.md",
        fm(
            name="router_prompt",
            version="2.0",
            owner="router",
            description="Router prompt with stronger follow-up handling",
            output_format="json",
            required=["question"],
            tags=["router", "candidate"],
            status="candidate",
            includes=["shared.system_rules"],
        )
        + """
You are an AI planner for a logistics operations assistant.

Do NOT answer the user.
ONLY classify capability and intent.

Capabilities: sql | rag | multi | unknown

Additional rules:
- Follow-ups that inherit entities/dates from conversation still map to sql/rag/multi (not unknown).
- Chart/dashboard requests are sql (visualization is planned later).
- File upload analysis requests are unknown for this legacy router (handled by IntentRouter elsewhere).

Return ONLY JSON:
{"capability":"sql","intent":"...","confidence":0.0,"reasoning":"..."}

Question:
{{question}}

Conversation memory:
{{memory}}
""",
    )
    write(
        PROMPTS / "router" / "registry.yaml",
        registry(
            "router",
            {
                "router_prompt": {
                    "active": "v1",
                    "candidates": ["v2"],
                    "experimental": [],
                    "versions": {
                        "v1": {"file": "router_prompt_v1.md", "status": "active"},
                        "v2": {"file": "router_prompt_v2.md", "status": "candidate"},
                    },
                }
            },
        ),
    )

    # planner
    write(
        PROMPTS / "planner" / "planner_prompt_v1.md",
        fm(
            name="planner_prompt",
            version="1.0",
            owner="planner",
            description="Creates multi-step execution plans",
            output_format="json",
            required=["question", "available_tools"],
            tags=["planner", "production"],
            includes=["shared.system_rules", "shared.business_rules"],
        )
        + """
You are an operations planning manager for GOFO logistics.
You NEVER answer the user's question.
You ONLY produce a structured multi-step execution plan.

Available tools: {{available_tools}}
Future tools may appear as plain strings.

Data source rules (minimum necessary):
- SQL: metrics, counts, KPIs, aggregations, rankings, historical data.
- KNOWLEDGE_GRAPH: relationships (Driver→Hub→Region, Manager→Hub, SOP ownership).
- RAG: unstructured SOP/policy/document explanations.
- MEMORY: follow-ups / conversation context.
- TRANSFORM: prepare/reshape DataFrames.
- STATISTICS: compute metrics.
- VISUALIZATION: only when charts/plots/dashboards are requested.
- RECOMMENDATION: business insights from computed results (never raw SQL).
- Prefer TRANSFORM→STATISTICS→VISUALIZATION→RECOMMENDATION over monolithic PYTHON.
- Combine sources only when necessary.

Rules:
- Break complex questions into small reliable steps.
- Minimize total tool calls while maximizing answer quality.
- Use depends_on for ordering (step numbers).
- For follow-ups, inherit date/entity context via step inputs.
- If ambiguous, set requires_clarification=true and clarification_question.
- Always end analytical plans with an LLM summarize step.
- Do not invent SQL or final answers.

Return ONLY JSON with this shape:
{
  "goal": "short goal",
  "selected_data_sources": ["SQL"],
  "source_reasons": {"SQL": "KPI metric question"},
  "expected_outputs": ["SQL: KPI rows"],
  "steps": [
    {
      "step_number": 1,
      "tool": "SQL",
      "action": "query_pickups",
      "description": "Retrieve pickup metrics",
      "inputs": {"question": "..."},
      "depends_on": []
    }
  ],
  "estimated_tool_calls": 2,
  "requires_clarification": false,
  "clarification_question": null,
  "confidence": 0.9,
  "reasoning": "debug only"
}

Question:
{{question}}

Memory:
{{memory}}

Intent classification:
{{intent_classification}}
""",
    )
    write(
        PROMPTS / "planner" / "planner_prompt_v2.md",
        fm(
            name="planner_prompt",
            version="2.0",
            owner="planner",
            description="Planner prompt with explicit retrieval_policy slot",
            output_format="json",
            required=["question", "available_tools"],
            tags=["planner", "candidate"],
            status="candidate",
            includes=["shared.system_rules", "shared.business_rules"],
        )
        + """
You are an operations planning manager for GOFO logistics.
You NEVER answer the user. You ONLY emit an ExecutionPlan JSON.

Available tools: {{available_tools}}

Additional v2 rules:
- When RAG is selected, include retrieval_policy:
  {"allow_general_knowledge": false, "minimum_confidence": "MEDIUM", "allow_clarification": true, "allow_document_request": true}
- Prefer MEMORY before SQL on follow-ups.
- Prefer STATISTICS over PYTHON alias.

Return ONLY JSON ExecutionPlan (same schema as v1) plus optional:
"retrieval_policy": {...}

Question:
{{question}}

Memory:
{{memory}}

Intent classification:
{{intent_classification}}
""",
    )
    write(
        PROMPTS / "planner" / "registry.yaml",
        registry(
            "planner",
            {
                "planner_prompt": {
                    "active": "v1",
                    "candidates": ["v2"],
                    "experimental": [],
                    "versions": {
                        "v1": {"file": "planner_prompt_v1.md"},
                        "v2": {"file": "planner_prompt_v2.md"},
                    },
                }
            },
        ),
    )

    # sql
    write(
        PROMPTS / "sql" / "generator_prompt_v1.md",
        fm(
            name="generator_prompt",
            version="1.0",
            owner="sql",
            description="Generate SQLite SQL only from retrieved schema",
            output_format="sql",
            required=["schema", "business_context", "few_shot_examples"],
            tags=["sql", "production"],
            includes=["shared.business_rules"],
            max_tokens=800,
        )
        + """
You are an expert SQLite analytics assistant.
You generate SQL ONLY.

Database engine:
SQLite

Retrieved schema (use ONLY these tables/columns):
{{schema}}

Business vocabulary:
{{business_context}}

Few-shot examples:
{{few_shot_examples}}

Hard rules:
- Output a single SQLite SELECT (or WITH ... SELECT).
- Never invent tables or columns.
- Prefer joins present in the retrieved schema relationships.
- If unanswerable from schema, return: SELECT 'UNKNOWN' AS answer;

Question:
{{question}}
""",
    )
    write(
        PROMPTS / "sql" / "generator_prompt_v2.md",
        fm(
            name="generator_prompt",
            version="2.0",
            owner="sql",
            description="SQL generator with stricter date and hub rules",
            output_format="sql",
            required=["schema", "business_context", "few_shot_examples"],
            tags=["sql", "candidate"],
            status="candidate",
            includes=["shared.business_rules"],
            max_tokens=800,
        )
        + """
You are an expert SQLite analytics assistant.
Generate SQL ONLY. Never explain.

Schema:
{{schema}}

Vocabulary:
{{business_context}}

Examples:
{{few_shot_examples}}

Stricter v2 rules:
- Always filter operational dates with pickups.pickup_date.
- Hub/warehouse always means drivers.hub.
- For rankings, include LIMIT unless the user asks for all rows.
- Unknown → SELECT 'UNKNOWN' AS answer;

Question:
{{question}}
""",
    )
    write(
        PROMPTS / "sql" / "planner_prompt_v1.md",
        fm(
            name="planner_prompt",
            version="1.0",
            owner="sql",
            description="SQL strategy notes before generation",
            required=["question"],
            tags=["sql", "strategy"],
        )
        + """
Produce a brief SQL strategy for the question (not SQL itself):
- target metric
- grain (driver/hub/day)
- filters / date range
- joins needed

Question:
{{question}}

Schema summary:
{{schema}}
""",
    )
    write(
        PROMPTS / "sql" / "validator_prompt_v1.md",
        fm(
            name="validator_prompt",
            version="1.0",
            owner="sql",
            description="Validate SQL against schema allowlist (LLM assist)",
            output_format="json",
            required=["sql", "schema"],
            tags=["sql", "validation"],
        )
        + """
Validate the SQL against the schema allowlist.
Return ONLY JSON:
{"valid": true, "issues": [], "rewritten_sql": null}

SQL:
{{sql}}

Schema:
{{schema}}
""",
    )
    write(
        PROMPTS / "sql" / "summarizer_prompt_v1.md",
        fm(
            name="summarizer_prompt",
            version="1.0",
            owner="sql",
            description="Summarize SQL rows into a business answer. Never generate SQL.",
            required=["question", "sql", "rows"],
            tags=["sql", "summarizer", "production"],
            includes=["shared.formatting_rules"],
            max_tokens=600,
        )
        + """
You are an operational analytics assistant.
Never generate SQL.
Never mention SQL.

The user asked:
{{question}}

The executed SQL was:
{{sql}}

The database returned:
{{rows}}

Write a concise business answer.
If there are no rows, reply exactly:
No matching operational records were found.
""",
    )
    write(
        PROMPTS / "sql" / "registry.yaml",
        registry(
            "sql",
            {
                "generator_prompt": {
                    "active": "v1",
                    "candidates": ["v2"],
                    "versions": {
                        "v1": {"file": "generator_prompt_v1.md"},
                        "v2": {"file": "generator_prompt_v2.md"},
                    },
                },
                "planner_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "planner_prompt_v1.md"}},
                },
                "validator_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "validator_prompt_v1.md"}},
                },
                "summarizer_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "summarizer_prompt_v1.md"}},
                },
            },
        ),
    )

    # rag
    write(
        PROMPTS / "rag" / "retrieval_prompt_v1.md",
        fm(
            name="retrieval_prompt",
            version="1.0",
            owner="rag",
            description="Rewrite questions for SOP semantic retrieval",
            required=["question"],
            tags=["rag", "retrieval", "production"],
            max_tokens=200,
        )
        + """
You rewrite user questions for semantic search over GOFO operations SOP documents.
Do NOT answer the question.
Preserve the original meaning.
Expand abbreviations when possible (for example, CBT -> Collection by TikTok (CBT)).
Make implicit subjects explicit and reference GOFO SOP context when helpful.
Keep the rewritten query concise as one clear question sentence.
Return only the rewritten question with no preamble or explanation.

Original question:
{{question}}

Rewritten question:
""",
    )
    write(
        PROMPTS / "rag" / "generator_prompt_v1.md",
        fm(
            name="generator_prompt",
            version="1.0",
            owner="rag",
            description="Grounded SOP answers from retrieved documents only",
            required=["question", "sop_context"],
            tags=["rag", "generator", "production"],
            includes=["shared.system_rules", "shared.formatting_rules"],
            max_tokens=900,
        )
        + """
You are the GOFO Operations Intelligence Assistant.
Answer ONLY using the SOP context provided below.
Never invent steps, policies, numbers, or names that are not in the context.
If the context does not contain enough information, respond exactly with:
I don't know based on the available SOP documents.
Be concise and operational.
Do not mention context, chunks, or retrieved documents in your answer.

Retrieval confidence: {{confidence_level}} ({{confidence_score}})
Reason: {{confidence_reason}}
Strategy: {{fallback_strategy}}

Question:
{{question}}

SOP Context:
{{sop_context}}
""",
    )
    write(
        PROMPTS / "rag" / "generator_prompt_v2.md",
        fm(
            name="generator_prompt",
            version="2.0",
            owner="rag",
            description="Cautious RAG generator with explicit uncertainty phrasing",
            required=["question", "sop_context"],
            tags=["rag", "generator", "candidate"],
            status="candidate",
            includes=["shared.system_rules", "shared.formatting_rules"],
            max_tokens=900,
        )
        + """
You are the GOFO Operations Intelligence Assistant.
Answer ONLY from SOP context.

If confidence is MEDIUM/LOW, open with:
"Based on the available SOP documentation..."
and note that additional procedures may exist.

If context is insufficient, reply exactly:
I don't know based on the available SOP documents.

Confidence: {{confidence_level}} / {{confidence_score}}
Strategy: {{fallback_strategy}}

Question:
{{question}}

SOP Context:
{{sop_context}}
""",
    )
    write(
        PROMPTS / "rag" / "confidence_prompt_v1.md",
        fm(
            name="confidence_prompt",
            version="1.0",
            owner="rag",
            description="Optional LLM assist for retrieval confidence rationale",
            output_format="json",
            required=["question", "similarity_scores"],
            tags=["rag", "confidence"],
            max_tokens=400,
        )
        + """
Evaluate whether retrieved evidence is sufficient for a grounded SOP answer.
Return ONLY JSON:
{"confidence_level":"HIGH|MEDIUM|LOW","reason":"...","should_clarify":false}

Question:
{{question}}

Similarity scores:
{{similarity_scores}}

Sources:
{{retrieved_sources}}

Chunk count:
{{retrieved_chunk_count}}
""",
    )
    write(
        PROMPTS / "rag" / "registry.yaml",
        registry(
            "rag",
            {
                "retrieval_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "retrieval_prompt_v1.md"}},
                },
                "generator_prompt": {
                    "active": "v1",
                    "candidates": ["v2"],
                    "versions": {
                        "v1": {"file": "generator_prompt_v1.md"},
                        "v2": {"file": "generator_prompt_v2.md"},
                    },
                },
                "confidence_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "confidence_prompt_v1.md"}},
                },
            },
        ),
    )

    # python / recommendation / reflection / clarification
    write(
        PROMPTS / "python" / "analysis_prompt_v1.md",
        fm(
            name="analysis_prompt",
            version="1.0",
            owner="python",
            description="Explain Python analytics outputs; never invent calculations",
            required=["question", "statistics"],
            tags=["python", "analytics"],
            includes=["shared.formatting_rules"],
        )
        + """
You are a GOFO analytics interpreter.
Use ONLY the provided statistics / tool outputs.
Never invent numbers.

Question:
{{question}}

Statistics:
{{statistics}}

Transformed preview:
{{dataframe_preview}}
""",
    )
    write(
        PROMPTS / "python" / "registry.yaml",
        registry(
            "python",
            {
                "analysis_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "analysis_prompt_v1.md"}},
                }
            },
        ),
    )

    write(
        PROMPTS / "recommendation" / "recommendation_prompt_v1.md",
        fm(
            name="recommendation_prompt",
            version="1.0",
            owner="recommendation",
            description="Generate operational recommendations from analytics",
            required=["question", "statistics"],
            tags=["recommendation", "production"],
            includes=["shared.business_rules", "shared.formatting_rules"],
            max_tokens=700,
        )
        + """
You are a GOFO operations advisor.
Generate practical operational recommendations from the analytics evidence.
Do not invent metrics that are not present.
Prefer actionable bullets (coach, rebalance, investigate root cause).

Question:
{{question}}

Statistics:
{{statistics}}

SQL summary:
{{sql_summary}}

Existing recommendations:
{{recommendations}}
""",
    )
    write(
        PROMPTS / "recommendation" / "registry.yaml",
        registry(
            "recommendation",
            {
                "recommendation_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "recommendation_prompt_v1.md"}},
                }
            },
        ),
    )

    write(
        PROMPTS / "reflection" / "reflection_prompt_v1.md",
        fm(
            name="reflection_prompt",
            version="1.0",
            owner="reflection",
            description="Critique draft answers; never rewrite the final answer",
            output_format="json",
            required=["question", "answer"],
            tags=["reflection", "production"],
            includes=["shared.system_rules"],
            max_tokens=800,
        )
        + """
You are a strict operations answer critic for GOFO logistics.
You NEVER write a replacement answer for the user.
You ONLY critique the draft answer.

Evaluate:
1. Did the answer directly answer the user's question?
2. Was enough evidence retrieved?
3. Are important facts missing?
4. Are there unsupported claims?
5. Are SQL results sufficient?
6. Is another retrieval likely to improve the answer?
7. Should another tool be executed?
8. Is confidence high enough to return this answer?

Return ONLY JSON:
{
  "approved": false,
  "confidence": 0.67,
  "should_retry_retrieval": true,
  "should_retry_sql": false,
  "should_retry_python": false,
  "should_ask_user": false,
  "missing_information": ["..."],
  "feedback": ["..."],
  "clarification_question": null,
  "reasoning": "debug only"
}

Question:
{{question}}

Draft answer:
{{answer}}

Evidence summary:
{{evidence}}
""",
    )
    write(
        PROMPTS / "reflection" / "registry.yaml",
        registry(
            "reflection",
            {
                "reflection_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "reflection_prompt_v1.md"}},
                }
            },
        ),
    )

    write(
        PROMPTS / "clarification" / "clarification_prompt_v1.md",
        fm(
            name="clarification_prompt",
            version="1.0",
            owner="clarification",
            description="Generate concise clarification questions with options",
            required=["question", "missing_fields"],
            tags=["clarification", "production"],
            includes=["shared.formatting_rules"],
            max_tokens=400,
        )
        + """
Generate ONE concise clarification question for a GOFO operations user.
Ask only for missing business parameters.
Prefer multiple-choice options when possible.
Do not answer the original question.

Original question:
{{question}}

Missing fields:
{{missing_fields}}

Ambiguity type:
{{ambiguity_type}}

Reason:
{{reason}}

Known options (if any):
{{options}}
""",
    )
    write(
        PROMPTS / "clarification" / "registry.yaml",
        registry(
            "clarification",
            {
                "clarification_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "clarification_prompt_v1.md"}},
                }
            },
        ),
    )

    # intent classifier (under router family alias file also useful)
    write(
        PROMPTS / "router" / "intent_classifier_prompt_v1.md",
        fm(
            name="intent_classifier_prompt",
            version="1.0",
            owner="router",
            description="Primary IntentType classifier JSON",
            output_format="json",
            required=["question", "allowed_intents"],
            tags=["router", "intent", "production"],
            includes=["shared.system_rules"],
        )
        + """
You are the intent classifier for a GOFO logistics operations intelligence agent.
Classify the user's latest message into exactly ONE primary intent.

Allowed intents: {{allowed_intents}}

Rules:
- Use conversation history for follow-ups.
- SOP/policy/procedure → SOP_QA / SOP_Summary / SOP_Compare.
- Simple operational lists/counts → SQL_Query.
- Analytical ops questions → SQL_Analysis.
- Charts/dashboards → Dashboard.
- Explaining prior numbers/charts → Explain_Result.
- File/CSV/PDF analysis → Upload_File.
- Coding/debug → Coding.
- Greetings → Greeting; casual chat → ChitChat; general non-ops knowledge → General_Knowledge.
- If unsure, use Unknown with low confidence.

Return ONLY JSON:
{
  "intent": "SQL_Analysis",
  "confidence": 0.96,
  "requires_sql": true,
  "requires_rag": false,
  "requires_memory": true,
  "requires_planner": true,
  "requires_clarification": false,
  "reasoning": "brief debug reason"
}

Question:
{{question}}

History:
{{history}}
""",
    )
    # update router registry to include intent classifier
    write(
        PROMPTS / "router" / "registry.yaml",
        registry(
            "router",
            {
                "router_prompt": {
                    "active": "v1",
                    "candidates": ["v2"],
                    "versions": {
                        "v1": {"file": "router_prompt_v1.md"},
                        "v2": {"file": "router_prompt_v2.md"},
                    },
                },
                "intent_classifier_prompt": {
                    "active": "v1",
                    "versions": {"v1": {"file": "intent_classifier_prompt_v1.md"}},
                },
            },
        ),
    )

    write(
        PROMPTS / "experiments.yaml",
        """
experiments:
  - id: exp_planner_prompt
    prompt_key: planner.planner_prompt
    versions: [v1, v2]
    description: Compare planner prompt versions
    priorities: [accuracy, sql_success, scenario_success, latency, cost]
    benchmark_limit: 40
    scenario_limit: 10
    mode: mock
  - id: exp_rag_generator
    prompt_key: rag.generator_prompt
    versions: [v1, v2]
    description: Compare RAG generator prompt versions
    priorities: [accuracy, scenario_success, latency, cost]
    benchmark_limit: 40
    scenario_limit: 8
    mode: mock
  - id: exp_sql_generator
    prompt_key: sql.generator_prompt
    versions: [v1, v2]
    description: Compare SQL generator prompt versions
    priorities: [sql_success, accuracy, latency, cost]
    benchmark_limit: 40
    scenario_limit: 8
    mode: mock
""",
    )
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
