# DEVELOPMENT_LOG

Append-only engineering history for `gofo_agent`.  
**Do not overwrite previous sessions.** Always append a new dated section at the bottom.

`README.md` describes the **current** project state. This file preserves **how we got there**.

---

# Session — 2026-07-19

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-19 (conversation spanned ~2026-07-05 → 2026-07-19) |
| **Branch** | `cursor-memory-version` (base: `main`) |
| **Overall objective** | Harden multimodal GOFO agent: attachment intelligence, centralized routing, hybrid RAG, ADA-style file analysis, fix Excel first-row truncation, generate charts with Python (matplotlib), keep durable docs before context-window reset |
| **Latest commit** | `2eb2622` — *Add attachment ADA analysis, hybrid RAG routing, and matplotlib Excel charts.* |

---

## Work Completed

### Features implemented

1. **Centralized Intent Router**
   - Routes: `GENERAL_CHAT`, `SOP_QA`, `SQL_ANALYTICS`, `ATTACHMENT_ANALYSIS`, `ATTACHMENT_VISUALIZATION`, `WAIT_FOR_UPLOAD`, `FOLLOW_UP`, `OPENAI_FALLBACK`
   - Attachment mode can activate/deactivate; SQL/chat detaches attachments so uploads no longer permanently hijack the session

2. **Hybrid RAG**
   - Dense Chroma + BM25 + Reciprocal Rank Fusion + cross-encoder rerank (`BAAI/bge-reranker-base`)
   - Ingest builds `bm25_corpus.json`; `retrieve()` API preserved

3. **Attachment / ADA pipeline**
   - Upload → validate → type-specific processors → `ProcessedFileContext`
   - Parse-once DataFrame store; per-prompt `analyze_dataframe` via `analysis_intent`
   - Streamlit preview modal (sheet selector for Excel)
   - Wait-for-upload replies when user asks for file analysis without an attachment

4. **Matplotlib chart generation (this session focus)**
   - Replaced Streamlit-native chart specs as primary visualization path
   - Server renders bar/line/pie/scatter/histogram/heatmap to PNG `image_base64`
   - Frontend displays via `st.image`
   - CJK font support (`WenQuanYi Zen Hei`) in Docker for Chinese logistics labels

5. **Conversational / production layers (earlier in conversation)**
   - Conversation resolver, semantic orchestration, intent classifier, context builders
   - Session-aware FastAPI; shared `AttachmentService` across session agents
   - Long-term SQLite memory, KPI/root-cause/anomaly/recommendations

### Bugs fixed

1. **Excel preview + analysis only saw the first data row**
   - Affected real GOFO exports (`pickup_driver_task_detail…xlsx`, `order_1784414170040.xlsx`)
   - Root cause: openpyxl `read_only=True` trusts incorrect worksheet dimensions (`max_row=2`)
   - Fix: `tools/files/excel_reader.py` uses non-read-only openpyxl + pandas fallback; preview and `excel_processor` both call it
   - Secondary issue: Docker containers were still running **stale image code** without source mounts → recreate with bind mounts

2. **Streamlit duplicate widget keys** on attachment Preview buttons when the same file appeared in multiple messages — fixed with message-index + attachment-index keys

3. **API “could not complete the agent request” on xlsx** — session agent used a separate stale attachment index; fixed by sharing `AttachmentService` in `api/server.py`

4. **Chart-by-系统单号 / one bar** — after only one row loaded, charts looked empty; also skip high-cardinality ID-like columns in categorical selection once rows are fixed

### Refactoring / architecture

- Agent path: `GOFOAgent` → `IntentRouter` → `RouteDispatcher`
- Shared Excel IO for UI preview and backend analysis
- Charts return structured payloads: `{type, title, renderer: matplotlib, format: png, image_base64, data}`

### UI improvements

- Bottom chat composer with “+” upload
- Attachment preview modal
- Visualizations section renders Python-generated PNG images

### Backend / Docker

- `docker-compose.yml` bind-mounts `api`, `core`, `tools`, `frontend`, config, assets, data, chroma
- `MPLCONFIGDIR=/tmp/matplotlib`
- Dockerfile installs `fonts-wqy-zenhei` + matplotlib in requirements
- Images rebuilt; fonts verified (`WenQuanYi Zen Hei`)

### Documentation

- Rewrote `README.md` to current-state source of truth
- Created this `DEVELOPMENT_LOG.md` session entry

### Memory / RAG / agent

- Extended `ConversationState` with `last_route`, `attachment_active`, attachment metadata
- Hybrid retrieval config in `config.py`
- ADA intents including visualization and wait-for-upload

---

## Files Created

| File | Purpose |
|------|---------|
| `core/intent_router.py` | Centralized route decisions |
| `core/route_dispatcher.py` | Execute chosen route handlers |
| `frontend/__init__.py` | Package marker for UI |
| `frontend/attachment_preview.py` | Attachment preview payloads/modal helpers |
| `frontend/attachment_ui.py` | Upload/history attachment UI |
| `tools/files/*` | Full attachment + ADA stack (reader, processors, analyzer, charts, store, memory, …) |
| `tools/files/excel_reader.py` | Robust Excel sheet reader (anti-truncation) |
| `tools/files/charts.py` | Matplotlib PNG chart builder |
| `tools/files/data_analysis.py` | Per-prompt DataFrame analysis |
| `tools/files/dataframe_store.py` | In-memory DataFrame cache |
| `tools/rag/hybrid.py` | Hybrid retrieval pipeline |
| `tools/rag/bm25_index.py` | BM25 index |
| `tools/rag/corpus.py` | Corpus helpers for BM25 |
| `tools/conversation/*` | Conversational resolver |
| `tools/context/*` | Context builders |
| `tools/orchestration/*` | Semantic orchestration |
| `tools/analyzer/*` | Business reasoner |
| `tools/planner/intent_classifier.py` | Intent classification |
| `tests/test_ada_attachment_analysis.py` | ADA + chart tests |
| `tests/test_excel_reader.py` | Excel truncation regressions |
| `tests/test_attachment_preview.py` | Preview tests |
| `tests/test_attachments.py` | Upload/processor tests |
| `tests/test_intent_router.py` | Router tests |
| `tests/test_hybrid_retriever.py` | Hybrid RAG tests |
| `tests/test_conversation_resolver.py` | Resolver tests |
| `tests/test_conversational_pipeline.py` | Pipeline tests |
| `tests/test_semantic_orchestration.py` | Orchestration tests |
| `DEVELOPMENT_LOG.md` | This history file |

---

## Files Modified

| File | Why |
|------|-----|
| `core/agent.py` | Wire IntentRouter, AttachmentMemory, dispatcher |
| `core/models.py` | Charts + richer response fields |
| `core/errors.py` / `error_handler.py` | `FileError` and messaging |
| `tools/memory/state.py` | Route/attachment state fields |
| `tools/rag/retriever.py` | Call hybrid path when enabled |
| `tools/router.py` | Align with conversational/attachment routing |
| `api/server.py` | Shared AttachmentService; upload/ask |
| `api/schemas.py` | `charts` on AskResponse; upload schemas |
| `frontend/app.py` | Composer, preview, `st.image` for matplotlib charts |
| `ingest.py` | Persist BM25 corpus alongside Chroma |
| `config.py` | Hybrid RAG + upload settings |
| `requirements.txt` | pandas, rank-bm25, sentence-transformers, matplotlib |
| `Dockerfile` | CJK fonts, `MPLCONFIGDIR` |
| `docker-compose.yml` | Source mounts, assets, matplotlib env |
| `README.md` | Full rewrite to current architecture |
| `.env.example` | Upload + hybrid-related vars |
| CLI / tests / planner / analysis context | Integration with new pipeline |

**Explicitly not committed:** `.env`, `data/uploads/*`, `chroma_db/*`, `logs/*`, `__pycache__/*`, `.DS_Store`.

---

## Problems Encountered

### Problem 1 — Excel only shows first row in preview and charts

**Root cause:** Logistics `.xlsx` files declare wrong dimensions. openpyxl `read_only` returns header + 1 data row. Docker was also serving an old image without source mounts, so local pandas fixes never reached the UI.

**Solution:** Shared non-read_only reader; mount source in Compose; `docker compose up -d --force-recreate`. Verified: pickup file → 11 rows; order file → 8238 rows.

**Lessons learned:** Always verify the **running container code**, not just the host tree. After compose volume changes, recreate containers. Prefer one shared reader for preview and analysis.

### Problem 2 — Matplotlib missing / Chinese glyphs as boxes in Docker

**Root cause:** Image lacked `matplotlib` until rebuild; slim image had no CJK fonts; `font.sans-serif` list alone did not prefer WenQuanYi until explicit selection + apt install.

**Solution:** Add matplotlib to requirements; install `fonts-wqy-zenhei` in Dockerfile; configure font manager to prefer available CJK fonts; set `MPLCONFIGDIR`.

**Lessons learned:** Charting in headless Docker needs Agg + writable MPLCONFIGDIR + explicit CJK fonts for GOFO Chinese columns.

### Problem 3 — Attachment API / session desync

**Root cause:** Upload used a global service; session `GOFOAgent` held a separate empty index.

**Solution:** Inject shared `AttachmentService` into session agents in `api/server.py`.

**Lessons learned:** Process-wide attachment store must be shared across session-scoped agents.

### Problem 4 — Long Docker rebuilds

**Root cause:** `sentence-transformers` / torch pull large wheels during image build.

**Solution:** Accept longer builds; use source mounts for code iteration without rebuild; rebuild when deps/Dockerfile change.

---

## Important Decisions

1. **IntentRouter owns attachment activate/detach** — prevents permanent attachment hijack.
2. **ADA = parse once, analyze per prompt** — no canned “Operational metric requires investigation” reuse for file answers.
3. **Matplotlib PNGs over Streamlit native charts** — true Python graphics, consistent pie/heatmap, API-portable.
4. **Never primary-path openpyxl `read_only`** for GOFO Excel exports.
5. **Hybrid RAG behind existing `retrieve()`** — callers stay stable.
6. **SQLite-only memory/analytics** — no Redis unless requested.
7. **Docs split:** README = current; DEVELOPMENT_LOG = append-only history.
8. **Do not commit secrets/uploads/runtime DBs.**

---

## Testing

### Tests performed

```bash
pytest tests/test_excel_reader.py tests/test_attachment_preview.py -q
pytest tests/test_ada_attachment_analysis.py -q
pytest tests/ -q
```

### Results

- Excel / preview focused tests: passed
- ADA attachment analysis: **10 passed**
- Full suite (2026-07-19): **`250 passed`**, 137 warnings (mostly `datetime.utcnow` deprecation)

### Runtime verification

- Docker: `read_excel_sheets` → 11 rows / 8238 rows on real uploads
- Docker: `build_charts` → `renderer=matplotlib`, non-empty `image_base64`
- Font: `WenQuanYi Zen Hei` after image rebuild with `fonts-wqy-zenhei`
- Commit + push: `2eb2622` → `origin/cursor-memory-version`

### Remaining issues

- Cosmetic matplotlib `findfont` weight warnings on WenQuanYi TTC (charts still render)
- Long-lived processes may cache old 1-row DataFrames until re-upload / new session
- First Docker build with sentence-transformers remains heavy

---

## Next Recommended Tasks

1. **Session cache invalidation** — when a new upload arrives (or content hash changes), drop stale `_processed_cache` / DataFrame store entries so users need not manually start a new chat.
2. **Chart UX for logistics columns** — prefer `城市`, `司机`, `揽收状态`, weight/package metrics; avoid ID columns entirely in auto-visualize.
3. **Replace `datetime.utcnow()`** with timezone-aware UTC across memory/models to clear deprecation warnings.
4. **Optional Plotly interactive charts** only if product asks; keep matplotlib as default PNG path.
5. **SQL safety / templates** for top operational KPIs (deterministic paths).
6. **Long-term memory retention** — dedupe findings, TTL for old conversations.
7. **CI** — GitHub Action running `pytest` on push to `cursor-memory-version`.
8. **Auth / deploy** only when moving beyond local Docker.

---

## Notes for Future Agents

- Preview and analysis **must** import `tools.files.excel_reader` — do not reintroduce inline `load_workbook(..., read_only=True)`.
- After editing `docker-compose.yml` volumes: `docker compose up -d --force-recreate`.
- After editing `requirements.txt` / `Dockerfile`: `docker compose build && docker compose up -d`.
- Chart payload contract: prefer `image_base64`; keep `data` for debugging/tests.
- Attachment preview keys: include message index — never key only on `attachment_id`.
- Real fixture files may live under `data/uploads/` locally; do not commit them.
- Prior conversation transcript (Cursor): `9911d240-eb0f-476c-8d8e-900dda334ceb` — this log supersedes chat memory for engineering continuity.

---

*End of session 2026-07-19.*

---

# Session — 2026-07-19 (Intent Classifier + Planning Layer)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-19 |
| **Branch** | `cursor-memory-version` |
| **Overall objective** | Add a dedicated Intent Classifier and multi-step Planning Layer as the first stages of the agent pipeline, separating planning from execution (no LangGraph) |
| **Tests** | `280 passed` |

---

## Work Completed

### Features implemented

1. **`core/intent_classifier.py`**
   - `IntentType` enum (Greeting, ChitChat, SOP_*, SQL_*, Dashboard, Follow_Up, Upload_File, Coding, Unknown, …)
   - `IntentClassification` structured output (confidence, tool flags, clarification, debug reasoning)
   - Heuristics for greetings/SOP/SQL/dashboard/upload/follow-ups; GPT fallback via `get_llm()`
   - Confidence < 0.5 → Unknown + clarification
   - `DEBUG` dumps for intent/tools/memory/planner flags

2. **`core/planner.py`**
   - `ExecutionStep` / `ExecutionPlan` / `ToolName` / `Planner`
   - Deterministic templates for common intents; LLM planner when no template
   - Never answers users; never executes tools
   - Clarification plans when confidence is low

3. **`core/plan_executor.py`**
   - Dependency-ordered tool registry (SQL, RAG, MEMORY, PYTHON, VISUALIZATION, ATTACHMENT, LLM)
   - Unknown future tools skipped safely
   - Reuses substantive SQL/RAG/attachment answers for summarize steps to avoid redundant LLM calls

4. **Agent wiring (`core/agent.py`)**
   - Pipeline: classify → IntentRouter (attachments) → plan → PlanExecutor (SQL_Analysis / Dashboard / SOP_Compare) or RouteDispatcher
   - Business `classifier_intent` preserved; new taxonomy in `intent_classification`
   - Flags: `INTENT_CLASSIFIER_ENABLED`, `PLANNER_ENABLED`, `DEBUG`

5. **Tests**
   - `tests/test_core_intent_classifier.py`
   - `tests/test_core_planner.py`
   - `tests/test_plan_executor.py`
   - `tests/conftest.py` disables `PLANNER_ENABLED` by default so legacy agent integration tests keep using RouteDispatcher

### Important decisions

- No LangGraph — modular stages only
- Keep `tools/planner/*` capability planner and `IntentRouter` attachment semantics
- Planner execution is feature-flagged; classification always enriches response metadata when enabled

### Remaining / next

- Opt more intents (Follow_Up, Upload_File) into PlanExecutor without regressing conversation repair
- Session cache invalidation for stale DataFrames (prior session recommendation)
- Chart defaults for Chinese logistics columns

---

*End of session 2026-07-19 (planning layer).*

---

# Session — 2026-07-19 (Quality Assurance Pipeline)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-19 |
| **Branch** | `cursor-memory-version` |
| **Overall objective** | Replace simple reflection with a multi-stage QA pipeline (evidence / reasoning / completeness + DecisionEngine) and bounded planner retries |
| **Tests** | `tests/test_quality_assurance.py` (13 passed) + prior suite green with QA disabled by default in conftest |

---

## Work Completed

1. **`core/quality_assurance.py`**
   - `EvidenceValidator`, `ReasoningValidator`, `CompletenessValidator`
   - `DecisionEngine` + `QualityReport` + `QAAction`
   - `QualityAssurancePipeline` with `register_validator()` for future validators
   - Never generates answers; never executes tools

2. **Planner retry**
   - `Planner.plan_retry(...)` builds additional SQL/RAG/Python/LLM steps from QA feedback

3. **Agent wiring**
   - After generation: QA evaluate → optional retry loop (max `QA_MAX_RETRIES`, default 2)
   - Clarification path when DecisionEngine asks the user
   - Response fields: `quality_report`, `qa_retry_count`

4. **Config**
   - `QUALITY_ASSURANCE_ENABLED`, `QA_MAX_RETRIES`, `QA_SCORE_THRESHOLD`, `QA_APPROVE_THRESHOLD`
   - Tests disable QA by default (same pattern as planner execution)

### Decisions

- No LangGraph — modular stages only; QA updates report state for the agent loop
- Decision priority: ask_user → approve (high scores) → accept_with_limits (retry cap) → retry_retrieval / retry_plan / retry_sql / retry_python

### Next

- Opt-in agent integration tests for successful QA retry with mocked tools
- Additional validators (citation, PII, SQL safety) via `register_validator`

---

*End of session 2026-07-19 (QA pipeline).*

---

# Session — 2026-07-19 (Reflection / Self-Critique)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-19 |
| **Branch** | `cursor-memory-version` |
| **Overall objective** | Add a dedicated Reflection (self-critique) layer that never generates answers and drives Planner retries |
| **Tests** | `tests/test_reflection.py` + QA suite green |

---

## Work Completed

1. **`core/reflection.py`**
   - `ReflectionResult` structured critique output
   - `ReflectionAgent.critique` / `critique_response`
   - Uses existing QA validators under the hood; optional LLM merge (`REFLECTION_USE_LLM`)
   - Never generates final answers; never executes tools

2. **Agent wiring**
   - Primary critic path: `REFLECTION_ENABLED` → critique → `Planner.plan_retry` → regenerate (max 2)
   - Falls back to direct QA loop only when reflection is disabled
   - Response fields: `reflection_result`, `reflection_retry_count`

3. **Config**
   - `REFLECTION_ENABLED`, `REFLECTION_USE_LLM`, `REFLECTION_MAX_RETRIES`
   - Tests disable reflection by default in conftest

### Decisions

- Reflection is the product-facing critic API; QA remains the multi-dimension validator engine
- No LangGraph — modular stages only

---

*End of session 2026-07-19 (reflection).*

---

# Session — 2026-07-19 (Tool Orchestrator)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-19 |
| **Branch** | `cursor-memory-version` |
| **Overall objective** | Add a Tool Orchestration layer that executes Planner plans with sequential/parallel waves, dependencies, retries, and AgentState |
| **Tests** | `tests/test_tool_orchestrator.py` + plan executor suite green |

---

## Work Completed

1. **`core/tool_orchestrator.py`**
   - `ToolRegistry`, `ToolExecutor`, `ToolOrchestrator`
   - `ExecutionContext`, `ExecutionResult` (per step), `AgentState`, `OrchestrationResult`
   - Dependency waves: independent steps run in parallel (`ThreadPoolExecutor`)
   - Transient error retry (once); non-retryable errors fail fast
   - Extensible `register_tool()` for Snowflake/etc. without changing orchestration logic
   - Never makes planning decisions

2. **Integration**
   - `PlanExecutor` delegates to `ToolOrchestrator` when enabled and using the default tool registry
   - Custom registries (tests/mocks) keep the legacy sequential path
   - Config: `TOOL_ORCHESTRATOR_ENABLED`, `TOOL_ORCHESTRATOR_PARALLEL`, `TOOL_ORCHESTRATOR_MAX_WORKERS`
   - `QueryResponse.agent_state` exposed in API analysis payload

### Decisions

- Planner = WHAT; Orchestrator = HOW
- No LangGraph — modular node-ready stages only
- Parallel workers do not mutate AgentState; main thread commits results

---

*End of session 2026-07-19 (tool orchestrator).*

---

# Session — 2026-07-20 (Documentation refresh + control-plane consolidation)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-20 |
| **Branch** | `new_feature_1` (tracks `origin/new_feature_1`; evolved from `cursor-memory-version` work) |
| **Latest commit on tip** | `4ad63a8` — *Add new feature* |
| **Overall objective** | Before context-window reset: rewrite README to current architecture and append a complete engineering history of this Cursor conversation so a new agent can continue without chat memory |
| **Test status at doc refresh** | **`311 passed`**, 137 warnings (`datetime.utcnow` deprecations) |

---

## Work Completed

This conversation delivered a full agent **control plane** on top of the existing GOFO RAG/SQL/attachment stack, then refreshed permanent docs.

### Features implemented

1. **Project review (read-only)**
   - Read README + DEVELOPMENT_LOG; inspected architecture before coding
   - Established baseline: production-style local/Docker agent with IntentRouter + attachments + hybrid RAG

2. **Intent Classifier (`core/intent_classifier.py`)**
   - `IntentType` enum (Greeting, ChitChat, SOP_*, SQL_*, Dashboard, Follow_Up, Upload_File, Coding, Unknown, …)
   - `IntentClassification` structured output (confidence, tool flags, clarification, debug reasoning)
   - Heuristics first (greetings, SOP, SQL, dashboard, upload, follow-ups); GPT fallback via `get_llm()`
   - Confidence &lt; 0.5 → Unknown + clarification
   - Conversation-aware follow-up detection using memory history

3. **Multi-step Planner (`core/planner.py`)**
   - `ExecutionStep` / `ExecutionPlan` / `ToolName` / `Planner`
   - Deterministic templates for common intents; LLM planner when no template
   - `plan_retry(...)` builds additional SQL/RAG/Python/LLM steps from critic feedback
   - Never answers users; never executes tools

4. **Plan Executor (`core/plan_executor.py`)**
   - Dependency-ordered tool registry wrapping existing SQL/RAG/ADA/chart/LLM handlers
   - Later adapted to delegate to ToolOrchestrator when enabled

5. **Quality Assurance (`core/quality_assurance.py`)**
   - `EvidenceValidator`, `ReasoningValidator`, `CompletenessValidator`
   - `DecisionEngine` → `QualityReport` / `QAAction`
   - Extensible `register_validator()` for future validators
   - Never generates answers; never executes tools

6. **Reflection / self-critique (`core/reflection.py`)**
   - `ReflectionResult` + `ReflectionAgent`
   - Primary product-facing critic; uses QA validators underneath
   - Optional LLM critique merge (`REFLECTION_USE_LLM`, default false)
   - Bounded retries via Planner (`REFLECTION_MAX_RETRIES`, default 2)

7. **Tool Orchestrator (`core/tool_orchestrator.py`)**
   - `ToolRegistry`, `ToolExecutor`, `ToolOrchestrator`
   - `ExecutionContext`, per-step `ExecutionResult`, `AgentState`, `OrchestrationResult`
   - Parallel dependency waves (`ThreadPoolExecutor`); transient-error retry once
   - Future tools via `register_tool()` without changing orchestration logic
   - Planner = WHAT; Orchestrator = HOW

8. **Agent wiring (`core/agent.py`)**
   - Pipeline: classify → IntentRouter (attachments) → plan → orchestrate/dispatch → reflect/QA → optional `plan_retry`
   - Business `classifier_intent` preserved for API compatibility; new taxonomy in `intent_classification`
   - Response fields: `execution_plan`, `quality_report`, `reflection_result`, `agent_state`, retry counts

9. **Config / env**
   - `DEBUG`, `INTENT_CLASSIFIER_ENABLED`, `PLANNER_ENABLED`
   - `QUALITY_ASSURANCE_*`, `REFLECTION_*`, `TOOL_ORCHESTRATOR_*`
   - Updated `.env.example`

10. **Tests**
    - `tests/test_core_intent_classifier.py`
    - `tests/test_core_planner.py`
    - `tests/test_plan_executor.py`
    - `tests/test_quality_assurance.py`
    - `tests/test_reflection.py`
    - `tests/test_tool_orchestrator.py`
    - `tests/conftest.py` disables planner/QA/reflection execution by default for legacy integration tests

11. **Documentation**
    - Incremental README/DEVELOPMENT_LOG updates during feature work
    - Full README rewrite (this session) to current branch state
    - This consolidation DEVELOPMENT_LOG entry

### Bugs fixed / hardening during integration

- Classifier GPT failures in sandboxed tests → safer try/except + broader SQL heuristics
- Overwriting business `classifier_intent` with new IntentType broke conversational tests → keep legacy Intent values on `classifier_intent`
- Planner/QA/Reflection enabling by default broke integration tests that call live OpenAI → conftest disables those flags; unit tests cover modules directly
- LLM summarize steps calling live API during plan execution → reuse substantive SQL/RAG/attachment answers for summarize actions when present
- Parallel orchestrator race on AgentState → workers return outputs; main thread commits state
- DecisionEngine forcing retrieval retries for SQL-only answers → fixed scoring/priority so SQL evidence does not incorrectly trigger RAG retry

### Refactoring / architecture

- Evolved from “router → one tool” to multi-stage control plane without introducing LangGraph
- Kept `IntentRouter` for attachment session semantics
- Kept `tools/planner/*` capability planner and business Intent enum (not deleted)
- PlanExecutor remains as adapter/fallback for mocked registries in tests

### UI / backend / Docker / RAG / memory

- No Streamlit redesign in this conversation (UI already had attachments/charts from prior work)
- Backend: agent pipeline + config flags only
- No Dockerfile changes in this conversation (prior session already had CJK fonts + mounts)
- Hybrid RAG unchanged functionally; still behind `retrieve()`
- Memory modules unchanged except being consumed by classifier/planner/reflection context

---

## Files Created

| File | Purpose |
|------|---------|
| `core/intent_classifier.py` | Primary intent classification |
| `core/planner.py` | Multi-step ExecutionPlan + retry planning |
| `core/plan_executor.py` | Tool execution adapter / legacy sequential runner |
| `core/quality_assurance.py` | Multi-stage QA validators + DecisionEngine |
| `core/reflection.py` | ReflectionAgent / ReflectionResult critic API |
| `core/tool_orchestrator.py` | Parallel/sequential tool orchestration + AgentState |
| `tests/test_core_intent_classifier.py` | Classifier unit tests |
| `tests/test_core_planner.py` | Planner unit tests |
| `tests/test_plan_executor.py` | PlanExecutor unit tests |
| `tests/test_quality_assurance.py` | QA unit tests |
| `tests/test_reflection.py` | Reflection unit tests |
| `tests/test_tool_orchestrator.py` | Orchestrator unit tests |

*(Earlier conversation sessions also created attachment/hybrid modules; those predate this control-plane work and remain documented in prior DEVELOPMENT_LOG entries.)*

---

## Files Modified

| File | Why |
|------|-----|
| `core/agent.py` | Wire classifier → planner → orchestrator/dispatcher → reflection/QA |
| `core/models.py` | Add plan/QA/reflection/agent_state response fields |
| `core/plan_executor.py` | Delegate to ToolOrchestrator when enabled |
| `config.py` | Feature flags for all new stages |
| `.env.example` | Document new env vars |
| `tests/conftest.py` | Disable planner/QA/reflection by default in agent integration tests |
| `README.md` | Full rewrite to current architecture (2026-07-20) |
| `DEVELOPMENT_LOG.md` | Append feature sessions + this consolidation entry |

---

## Problems Encountered

### Problem 1 — New pipeline broke existing agent integration tests

**Root Cause:** Classifier/planner/executor paths called live OpenAI; also overwrote `analysis.intent` with new IntentType strings (`SQL_Analysis` vs `ROOT_CAUSE`).

**Solution:** Safer classifier fallbacks; preserve business Intent on `classifier_intent`; conftest disables planner/QA/reflection execution for legacy tests; dedicated unit tests with mocks.

**Lessons learned:** When adding a new taxonomy, keep the public API field stable or dual-write. Feature flags + conftest defaults protect the suite.

### Problem 2 — Parallel orchestration mutated shared state

**Root Cause:** ThreadPool workers updated `ExecutionContext` / `AgentState` concurrently.

**Solution:** `commit_state=False` in workers; main thread commits results after futures complete.

**Lessons learned:** Parallel tool waves need immutable-ish inputs and main-thread state merges.

### Problem 3 — QA DecisionEngine over-triggered retrieval retries

**Root Cause:** Low evidence score automatically set `should_retry_retrieval` even for SQL-only answers.

**Solution:** Only retry retrieval when evidence validator requests it; improve SQL numeric grounding scores; prefer SQL/plan retries for analytical gaps.

**Lessons learned:** Critic actions must be capability-aware.

### Problem 4 — Branch rename / tip vs docs drift

**Root Cause:** Work began on `cursor-memory-version` docs narrative; tip is now `new_feature_1` with commit `4ad63a8`.

**Solution:** This README rewrite documents `new_feature_1` as current branch and keeps historical DEVELOPMENT_LOG entries intact.

**Lessons learned:** Always re-check `git branch` / `git log` before permanent docs refresh.

---

## Important Decisions

1. **No LangGraph** — modular stages only; “LangGraph integration” means clean boundaries for a future wrapper.
2. **Do not delete IntentRouter / tools.planner** — attachment semantics and business Intent remain.
3. **Reflection is primary critic; QA is scoring engine** — avoid two competing retry loops in production path.
4. **Planner never executes; Orchestrator never plans.**
5. **Max 2 critic retries** — always return best-effort after limit.
6. **Tests disable heavy control-plane flags by default** — unit tests cover new modules with mocks.
7. **Excel `read_only` ban remains** — unchanged from prior sessions.
8. **README = current; DEVELOPMENT_LOG = append-only** — this entry consolidates chat history for the next agent.

---

## Testing

### Tests performed

```bash
export PYTHONPATH=.
pytest tests/test_core_intent_classifier.py tests/test_core_planner.py tests/test_plan_executor.py -q
pytest tests/test_quality_assurance.py tests/test_reflection.py -q
pytest tests/test_tool_orchestrator.py tests/test_plan_executor.py -q
pytest tests/ -q
```

### Results

- Focused module suites: passed during implementation
- Full suite at documentation refresh (2026-07-20): **`311 passed`**, 137 warnings (mostly `datetime.utcnow` deprecation)

### Remaining issues

- Stale attachment/DataFrame cache invalidation on re-upload still open (prior roadmap)
- Chart defaults for Chinese logistics columns still open
- `datetime.utcnow()` deprecations across memory modules
- Broadening orchestrator coverage for Follow_Up / Upload_File without regressing conversation repair
- Commit message `Add new feature` is vague; consider clearer commits next session

---

## Next Recommended Tasks

1. **Session cache invalidation** — when a new upload arrives or content hash changes, drop stale `_processed_cache` / DataFrame store entries.
2. **Chart UX for logistics columns** — prefer `城市` / `司机` / `揽收状态` / weight metrics; avoid ID-like columns in auto-visualize.
3. **Agent integration tests for Reflection retry** — enable `REFLECTION_ENABLED` in a focused test with mocked tools to prove approve/retry/ask_user end-to-end.
4. **Replace `datetime.utcnow()`** with timezone-aware UTC across memory/models.
5. **SQL safety / templates** for top operational KPIs (deterministic paths).
6. **CI** — GitHub Action running `pytest` on push to `new_feature_1`.
7. **Optional:** rename/clarify next commits; avoid committing uploads/chroma/logs.

---

## Notes for Future Agents

- Start with **README.md**, then the **latest DEVELOPMENT_LOG sessions** (2026-07-19 control plane + this 2026-07-20 consolidation).
- Current branch is **`new_feature_1`**, not necessarily `cursor-memory-version`.
- Control-plane files live under `core/`: `intent_classifier.py`, `planner.py`, `plan_executor.py`, `tool_orchestrator.py`, `quality_assurance.py`, `reflection.py`.
- Do **not** recreate those modules; extend them.
- Preview/analysis Excel must keep using `tools.files.excel_reader`.
- `tests/conftest.py` intentionally sets `PLANNER_ENABLED`, `QUALITY_ASSURANCE_ENABLED`, and `REFLECTION_ENABLED` to false for most agent integration tests.
- Prior attachment/hybrid/matplotlib work is still valid and must be preserved.
- This documentation supersedes ephemeral Cursor chat memory for engineering continuity.

---

*End of session 2026-07-20 (documentation refresh + control-plane consolidation).*

---

# Session — 2026-07-20 (Data Source Selection + Knowledge Graph)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-20 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Improve Planner to choose the minimum necessary data sources before Tool Orchestration (SQL / Knowledge Graph / RAG / Memory / Python) |
| **Test status** | **`321 passed`** |

---

## Work Completed

### Features implemented

1. **Data Source Selection layer** (`core/data_source_selector.py`)
   - `DataSource`, `SelectedSource`, `DataSourceSelection`, `DataSourceSelector`
   - Rules: SQL for metrics/KPIs; KG for relationships; RAG for SOPs/explanations; Memory for follow-ups; Python only for calc/stats/forecast/charts
   - Combines sources only when necessary
   - Never executes tools; never answers the user

2. **Knowledge Graph tool** (`tools/knowledge_graph/`)
   - Driver → Hub → Region (SQLite drivers + hub/region map)
   - Manager → Hub (demo org map)
   - SOP ownership map
   - Demo alias `Driver John` → `Logan Clark` (Chicago / Midwest)
   - Consumes prior SQL rows when resolving “who manages lowest-rate hub”

3. **Planner rewired**
   - `DATA_SOURCE_SELECTION_ENABLED` (default true)
   - Builds `ExecutionPlan` from selection: `selected_data_sources`, `source_reasons`, `expected_outputs`, `data_source_selection`
   - `ToolName.KNOWLEDGE_GRAPH` added
   - Stops always adding Python for SQL_Analysis rankings

4. **Execution wiring**
   - `_handle_knowledge_graph` in `plan_executor.py`
   - Registered in ToolOrchestrator default registry (+ KG aliases)
   - Agent planner intents expanded to SQL_Query / SOP_QA / SOP_Summary / Follow_Up / Explain_Result

5. **Tests**
   - `tests/test_data_source_selector.py` — all user examples
   - `tests/test_knowledge_graph.py`
   - Updated `tests/test_core_planner.py` for minimum-tool behavior

### Example plans (verified in tests)

| Question | Sources |
|----------|---------|
| What is today's pickup rate? | SQL only |
| Which region does Driver John belong to? | Knowledge Graph only |
| Explain the pickup SOP. | RAG only |
| Compare Chicago's pickup rate with yesterday and explain possible reasons. | SQL + RAG |
| Who manages the hub with the lowest pickup rate? | SQL + Knowledge Graph |
| Compare Midwest performance and generate a trend chart. | SQL + Python + Visualization |

---

## Files Created

| File | Purpose |
|------|---------|
| `core/data_source_selector.py` | Data Source Selection layer |
| `tools/knowledge_graph/__init__.py` | KG package exports |
| `tools/knowledge_graph/service.py` | Relationship query service |
| `tests/test_data_source_selector.py` | Selection + planner plan tests |
| `tests/test_knowledge_graph.py` | KG unit tests |

---

## Files Modified

| File | Why |
|------|-----|
| `core/planner.py` | Selection-first planning; plan metadata; KG tool name |
| `core/plan_executor.py` | KG handler + evidence reuse |
| `core/tool_orchestrator.py` | Register KG; store KG facts in AgentState |
| `core/agent.py` | Broader planner intent coverage |
| `config.py` / `.env.example` | `DATA_SOURCE_SELECTION_ENABLED` |
| `tests/test_core_planner.py` | Ranking no longer requires Python |
| `README.md` | Current architecture with source selection |
| `DEVELOPMENT_LOG.md` | This session |

---

## Problems Encountered

### Problem 1 — Ranking plans previously always called Python

**Root Cause:** SQL_Analysis templates always appended a PYTHON derive-KPI step.

**Solution:** Data Source Selection only adds PYTHON for calculate/statistics/forecast/chart signals; rankings stay SQL (+ LLM).

**Lessons learned:** Tool presence should be driven by question need, not intent label alone.

### Problem 2 — No real Knowledge Graph existed

**Root Cause:** Relationships were previously answered (poorly) via SQL/LLM only.

**Solution:** Lightweight KG service over drivers SQLite + static hub/region/manager/SOP maps; Planner selects it explicitly.

**Lessons learned:** Keep KG as a first-class tool so selection can choose it without inventing SQL joins for org structure.

---

## Important Decisions

1. **DataSourceSelector runs before plan step construction** — Planner consumes selection; Orchestrator still does not plan.
2. **Minimum necessary sources** — do not fan out to every tool.
3. **KG depends on SQL when both selected** — supports “lowest rate hub → manager”.
4. **Python is not the default for analytics** — only calc/stats/forecast/charts.
5. **`DATA_SOURCE_SELECTION_ENABLED=true` by default**; legacy templates remain as fallback when disabled.

---

## Testing

```bash
export PYTHONPATH=.
pytest tests/test_data_source_selector.py tests/test_knowledge_graph.py tests/test_core_planner.py -q
pytest tests/ -q
```

**Results:** focused suites passed; full suite **`321 passed`**, 137 warnings (`datetime.utcnow`).

**Remaining issues:** expand real org KG beyond demo maps; optional LLM-assisted source selection for ambiguous multi-domain questions.

---

## Next Recommended Tasks

1. Persist richer org graph (managers, regions) in SQLite instead of static maps.
2. Surface `selected_data_sources` / `source_reasons` in Streamlit debug panel.
3. Add integration test where Reflection retries change source selection.
4. Continue prior roadmap: stale attachment cache invalidation; `datetime.utcnow` cleanup; CI.

---

## Notes for Future Agents

- Read `core/data_source_selector.py` before changing planner templates.
- Do not reintroduce “always call Python for SQL_Analysis”.
- Extend `tools/knowledge_graph/service.py` for new relationship types; register via existing ToolRegistry.
- ExecutionPlan fields `selected_data_sources`, `source_reasons`, `expected_outputs` are part of the public plan contract.

---

*End of session 2026-07-20 (Data Source Selection + Knowledge Graph).*

---

# Session — 2026-07-22 (SQL Schema Retriever)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Retrieve only relevant tables/columns/joins before SQL generation instead of dumping the full schema into the LLM prompt |
| **Test status** | **`332 passed`** |

---

## Work Completed

### Features implemented

1. **Schema Registry** (`tools/sql/schema_registry.py`)
   - Live load via SQLite `PRAGMA table_info` / `foreign_key_list`
   - Stores tables, columns, PKs, FKs, curated descriptions/keywords
   - Auto-refresh when DB mtime changes; `refresh_schema_registry()` for explicit rebuild

2. **Schema Retriever** (`tools/sql/schema_retriever.py`)
   - Input: question + business intent / semantic context
   - Output: `RetrievedSchema` with tables, columns, relationships, candidates, scores
   - Strategy: business intent hints + keyword/description matching + optional embedder hook
   - Never returns full schema unless explicitly requested (“full schema”, “all tables”, …)

3. **SQL pipeline rewired** (`tools/sql/service.py`)
   - Question → Business Understanding → Schema Retriever → Generator → Validator → Executor
   - Generator (`planner.plan`) uses **only** retrieved schema in the system prompt
   - Validator (`tools/sql/validator.py`) rejects invented tables/columns

4. **AgentState / QueryResponse**
   - `retrieved_schema`, `candidate_tables`, `candidate_columns` stored on AgentState + API response
   - `_handle_sql` / orchestrator `_update_agent_state` wired

5. **Debug**
   - `DEBUG` or `SCHEMA_RETRIEVER_DEBUG` prints intent, tables, columns, joins, scores

6. **Tests**
   - `tests/test_schema_retriever.py` + updated SQL planner/service tests

---

## Files Created

| File | Purpose |
|------|---------|
| `tools/sql/schema_registry.py` | Schema Registry + refresh |
| `tools/sql/schema_retriever.py` | Relevant schema retrieval |
| `tools/sql/validator.py` | SQL allowlist validation |
| `tests/test_schema_retriever.py` | Registry/retriever/validator tests |

## Files Modified

| File | Why |
|------|-----|
| `tools/sql/planner.py` | Generate SQL from retrieved schema only |
| `tools/sql/service.py` | Full schema-aware pipeline |
| `tools/sql/schema_loader.py` | Prefer registry text helper |
| `tools/sql/__init__.py` | Export new APIs |
| `core/models.py` | Response fields for retrieved schema |
| `core/tool_orchestrator.py` | AgentState schema fields |
| `core/plan_executor.py` | Pass through schema metadata from SQL tool |
| `config.py` / `.env.example` | Schema retriever flags |
| `tests/test_sql_*.py` | Prompt assertions updated |
| `README.md` / `DEVELOPMENT_LOG.md` | Current docs |

---

## Important Decisions

1. **Never dump full schema by default** — replaces the previous triple schema injection.
2. **Join expansion is conservative** — adding a satellite table pulls in `pickups` for joins; selecting `pickups` alone does **not** pull every FK parent.
3. **Validator uses full registry allowlist** — retrieval can be narrow; validation still rejects unknown objects against the live DB.
4. **Embedding retrieval is optional/off by default** — `SchemaEmbedder` protocol is ready for future vector search.

---

## Testing

```bash
export PYTHONPATH=.
pytest tests/test_schema_retriever.py tests/test_sql_planner.py tests/test_sql_service.py -q
pytest tests/ -q
```

**Results:** **`332 passed`**, 137 warnings.

---

## Next Recommended Tasks

1. Implement OpenAI embedding `SchemaEmbedder` behind `SCHEMA_EMBEDDING_RETRIEVAL_ENABLED`.
2. Surface retrieved schema in Streamlit debug UI.
3. Persist SQLite `COMMENT` metadata if warehouses add column comments.
4. Prior roadmap: attachment cache invalidation, `datetime.utcnow` cleanup, CI.

---

## Notes for Future Agents

- Do not reintroduce full `SCHEMA` + live schema + hardcoded schema triple dumps into `planner._build_system_prompt`.
- Extend descriptions/keywords in `schema_registry.py` when adding tables — retrieval picks them up automatically after refresh.
- Call `refresh_schema_registry()` after DB migrations.

---

*End of session 2026-07-22 (SQL Schema Retriever).*

---

# Session — 2026-07-22 (Specialized Python Analytics Tools)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Replace monolithic Python analytics with specialized tools (Transform / Statistics / Visualization / Recommendation) |
| **Test status** | **`339 passed`** |

---

## Work Completed

### Features implemented

1. **`tools/python/transformation_tool.py`** — filter/sort/groupby/pivot/rename/fillna/date conversion
2. **`tools/python/statistics_tool.py`** — mean/median/min/max, percentages, ratios, growth, ranking, correlation
3. **`tools/python/visualization_tool.py`** — wraps matplotlib `build_charts`; returns charts + chart_metadata
4. **`tools/python/recommendation_tool.py`** — business insights from statistics (never raw SQL)

### Planner routing

```text
SQL → TRANSFORM → STATISTICS → VISUALIZATION (if requested) → RECOMMENDATION (if requested) → LLM
```

- DataSourceSelector emits TRANSFORM+STATISTICS instead of monolithic PYTHON
- `ToolName.PYTHON` kept as backward-compatible alias → StatisticsTool
- Dashboard template updated to specialized tool chain

### AgentState

Stores: `dataframe`, `transformed_dataframe`, `statistics`, `chart_metadata`, `recommendations`

### Debug

`DEBUG` or `PYTHON_TOOLS_DEBUG` prints tool name, functions executed, shape, stats/charts/recommendations, execution time.

---

## Files Created

| File | Purpose |
|------|---------|
| `tools/python/__init__.py` | Package exports |
| `tools/python/transformation_tool.py` | DataFrame prep |
| `tools/python/statistics_tool.py` | Metric computation |
| `tools/python/visualization_tool.py` | Charts |
| `tools/python/recommendation_tool.py` | Business recommendations |
| `tests/test_python_tools.py` | Unit + planner routing tests |

## Files Modified

| File | Why |
|------|-----|
| `core/planner.py` | New ToolNames + dependency order + dashboard template |
| `core/data_source_selector.py` | Select specialized Python tools |
| `core/plan_executor.py` | Handlers for TRANSFORM/STATISTICS/VISUALIZATION/RECOMMENDATION |
| `core/tool_orchestrator.py` | AgentState fields + registry aliases |
| `config.py` / `.env.example` | `PYTHON_TOOLS_DEBUG` |
| `tests/test_data_source_selector.py` | Chart plan expectations |
| `README.md` / `DEVELOPMENT_LOG.md` | Docs |

---

## Important Decisions

1. **LLM never calculates** — only explains Python tool outputs.
2. **Single responsibility per tool** — easy to extend with future analytics tools.
3. **Keep `PYTHON` alias** — existing plans/tests that emit PYTHON still work.
4. **Recommendation only when requested / dashboard / chart analytics path** — avoid forcing it onto every SQL+KG question.

---

## Testing

```bash
export PYTHONPATH=.
pytest tests/test_python_tools.py tests/test_data_source_selector.py -q
pytest tests/ -q
```

**Results:** **`339 passed`**

---

## Next Recommended Tasks

1. Surface `statistics` / `chart_metadata` / `recommendations` in Streamlit debug panel.
2. Add forecast tool under `tools/python/` when product requests forecasting.
3. Continue prior roadmap: attachment cache invalidation, `datetime.utcnow` cleanup, CI.

---

## Notes for Future Agents

- Do not recreate a monolithic `analytics.py`.
- Extend by adding new files under `tools/python/` and registering handlers in `DEFAULT_TOOL_REGISTRY`.
- Visualization still reuses `tools.files.charts.build_charts` for PNG rendering.

---

*End of session 2026-07-22 (Specialized Python Analytics Tools).*

---

# Session — 2026-07-22 (Clarification Manager)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Prevent hallucinations by asking for missing business parameters after Planner and before Tool Orchestrator; resume the original plan after the user answers |
| **Test status** | **`350 passed`** |

---

## Work Completed

### Features implemented

1. **`core/clarification_manager.py`**
   - Detects ambiguous ops questions (best driver metric, performance scope, compare periods, top customers metric, short ranking without time)
   - Multiple-choice options; accepts option number or label
   - `PendingClarification` stores `original_question`, `missing_fields`, `pending_question`, `user_response`, draft plan
   - `apply_user_response()` enriches the question and returns a resume decision
   - Skips when conversation memory already fills the slot or confidence + question are rich
   - Debug via `DEBUG` / `CLARIFICATION_DEBUG`

2. **Agent pipeline wiring** (`core/agent.py`)
   - Pipeline: Classifier → Planner → **ClarificationManager** → Tool Orchestrator
   - Session-scoped `pending_clarification` on `GOFOAgent`
   - Next turn resumes with enriched `resolved_question`; filled metric/date slots survive `memory.add_turn`
   - Reflection ask-user also seeds pending clarification

3. **Models / config**
   - `QueryResponse`: `requires_clarification`, `clarification_question`, `clarification_options`, `missing_fields`
   - `AgentState`: `original_question`, `missing_fields`, `pending_question`, `user_response`
   - `CLARIFICATION_MANAGER_ENABLED` (default true), `CLARIFICATION_DEBUG`

4. **Tests**
   - `tests/test_clarification_manager.py` — detection, memory skip, resume, agent ask→answer→execute
   - Conftest disables Clarification Manager by default for unrelated agent tests

---

## Files Created

| File | Purpose |
|------|---------|
| `core/clarification_manager.py` | Clarification Manager |
| `tests/test_clarification_manager.py` | Unit + agent resume tests |

## Files Modified

| File | Why |
|------|-----|
| `core/agent.py` | Wire evaluate/resume; persist slots on response |
| `core/models.py` | Clarification response fields |
| `core/tool_orchestrator.py` | AgentState clarification fields |
| `tools/memory/conversation.py` | Prefer `business_metric` in `_infer_metric` |
| `config.py` / `.env.example` | Feature flags |
| `tests/conftest.py` | Disable clarification by default in agent tests |
| `README.md` / `DEVELOPMENT_LOG.md` | Docs |

---

## Important Decisions

1. **Never guess missing business parameters** — ask only when required.
2. **Pending state lives on the session agent**, not ephemeral orchestrator state, so multi-turn resume works.
3. **Skip when memory already has the slot** — explicit skip reason for debug.
4. **Resume does not restart the conversation** — same draft plan context, enriched question.

---

## Testing

```bash
export PYTHONPATH=.
pytest tests/test_clarification_manager.py -q
pytest tests/ -q
```

**Results:** **`350 passed`**, 139 warnings.

---

## Next Recommended Tasks

1. Surface clarification options as clickable chips in Streamlit.
2. Expand patterns (custom date range picker, region filters).
3. Continue prior roadmap: attachment cache invalidation, `datetime.utcnow` cleanup, CI.

---

## Notes for Future Agents

- Extend ambiguity patterns in `_detect_ambiguity`; keep memory-fill logic in `evaluate()`.
- Do not re-enable Clarification Manager in unrelated agent tests without mocking PlanExecutor.
- After user answers, set `response.business_metric` / `date_range` so memory does not overwrite clarification slots.

---

*End of session 2026-07-22 (Clarification Manager).*

---

# Session — 2026-07-22 (Adaptive Retrieval Confidence Engine)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Evaluate retrieval quality with multi-signal adaptive scoring before generation; never answer confidently when evidence is weak; Planner controls fallbacks |
| **Test status** | **`363 passed`** |

---

## Work Completed

### Features implemented

1. **`tools/rag/confidence.py`**
   - Weighted signals: highest similarity 40%, average 30%, chunk count 15%, source diversity 10%, metadata quality 5%
   - Levels: HIGH ≥0.80 / MEDIUM 0.60–0.79 / LOW &lt;0.60
   - Decision Engine: GENERATE_CONFIDENT / GENERATE_CAUTIOUS / CLARIFY / DOCUMENT_REQUEST / GENERAL_KNOWLEDGE / REFUSE
   - `RetrievalPolicy` from Planner (`allow_general_knowledge`, `minimum_confidence`, `allow_clarification`, `allow_document_request`)
   - Tunable weights via `RETRIEVAL_CONF_W_*` (normalized at runtime)

2. **RAG service / generator**
   - Pipeline: Retriever → Confidence → Decision → Generator
   - Removed hard `SIMILARITY_THRESHOLD` generation gate
   - Generator receives confidence context and adapts prompts (cautious / general-knowledge disclaimer)

3. **Planner + PlanExecutor**
   - `ExecutionPlan.retrieval_policy` injected into RAG/LLM steps
   - SOP templates use strict policy (no general knowledge by default)
   - `_handle_llm` SOP summarize path uses confidence + `generate()`

4. **AgentState / QueryResponse**
   - Stores confidence_score/level/breakdown, similarity_scores, sources, reason, fallback_strategy, policy

5. **Tests**
   - `tests/test_retrieval_confidence.py` + updated `test_service.py` / `test_generator.py`

---

## Files Created

| File | Purpose |
|------|---------|
| `tools/rag/confidence.py` | Adaptive confidence + decision engine |
| `tests/test_retrieval_confidence.py` | Unit + planner policy tests |

## Files Modified

| File | Why |
|------|-----|
| `tools/rag/service.py` | Confidence before generate |
| `tools/rag/generator.py` | Confidence-aware prompts |
| `tools/rag/__init__.py` | Export confidence APIs |
| `core/planner.py` | `retrieval_policy` on plans |
| `core/plan_executor.py` | Wire policy + SOP grounded generate |
| `core/tool_orchestrator.py` | AgentState confidence fields |
| `core/models.py` | Response confidence fields |
| `config.py` / `.env.example` | Flags + weight knobs |
| `README.md` / `DEVELOPMENT_LOG.md` | Docs |

---

## Important Decisions

1. **No single similarity threshold** as a generation gate — adaptive multi-signal score only.
2. **LOW never produces a confident SOP answer** — clarify → document request → general knowledge (Planner only) → refuse.
3. **General knowledge must be explicitly permitted** and always carries the disclaimer.
4. **Weights live in config** so tuning does not change business logic.

---

## Testing

```bash
export PYTHONPATH=.
pytest tests/test_retrieval_confidence.py tests/test_service.py tests/test_generator.py -q
pytest tests/ -q
```

**Results:** **`363 passed`**, 139 warnings.

---

## Notes for Future Agents

- Extend scoring in `evaluate_retrieval`; keep Decision Engine policy-driven.
- Do not reintroduce a hard `SIMILARITY_THRESHOLD` skip before generate.
- When adding new document collections, reuse `evaluate_and_decide` — it is collection-agnostic.

---

*End of session 2026-07-22 (Adaptive Retrieval Confidence Engine).*

---

# Session — 2026-07-22 (Evaluation & Scenario Benchmark Framework)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Add a continuous evaluation framework for benchmark questions + multi-turn scenarios (accuracy, tools, latency, cost, reliability) with markdown reports and regression compare |
| **Test status** | framework tests green; full suite pending |

---

## Work Completed

1. **`evaluation/` package**
   - Datasets: 200 SOP / 100 SQL / 50 general / 50 follow-up / 30 memory / 20 upload / 40 scenarios
   - `runner.py`, `scenario_runner.py`, `evaluator.py`, `metrics.py`, `report.py`, `compare.py`
   - `agent_adapter.py` with live + mock modes
   - One command: `python -m evaluation`

2. **Scoring**
   - Semantic keyword proxy (no exact answer match)
   - Tool selection/order, behavior match, SQL success, retrieval confidence
   - Latency, tokens, estimated cost
   - Suite + overall agent scorecard

3. **Artifacts**
   - JSON under `evaluation/results/`
   - `latest_report.md` with executive summary, failures, regressions, recommendations

---

## How to run

```bash
export PYTHONPATH=.
python -m evaluation --generate-datasets
python -m evaluation --mode mock
python -m evaluation --mode live --limit 20
pytest tests/test_evaluation_framework.py -q
```

---

## Notes for Future Agents

- Add questions to `evaluation/datasets/*.json` or regenerate via `generate_datasets.py`.
- Prefer `--mode live` for release gating; mock mode is for offline smoke/CI of the framework itself.
- Extend metrics in `metrics.py` / `evaluator.py` without changing runners.

---

*End of session 2026-07-22 (Evaluation Framework).*

---

# Session — 2026-07-22 (Enterprise Prompt Registry & Experiment Framework)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-22 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Replace monolithic/inline prompts with a centralized versioned Prompt Registry as the single source of truth for LLM interactions (discovery, metadata, rendering, config, experiments) |
| **Test status** | **`375 passed`** (full suite) |

---

## Work Completed

1. **Core modules**
   - `core/prompt_validator.py` — YAML frontmatter → `PromptMetadata`
   - `core/prompt_renderer.py` — `{{variable}}` rendering + required-variable checks
   - `core/prompt_registry.py` — discover/load/cache, active/candidate/experimental, `PROMPT_EXPERIMENT`, hot reload, persist active
   - `core/prompt_manager.py` — `get` / `render` / `build_messages` / `invoke` / `llm_for` (metadata → LLM client)
   - `core/prompt_experiments.py` — run eval per version, recommend by priorities, write reports

2. **Prompt assets** under `prompts/` (router, planner, sql, rag, python, recommendation, reflection, clarification, shared) with `registry.yaml` + `experiments.yaml`

3. **Wiring** — Planner, intent classifier, reflection, router planner, SQL generator/summarizer, RAG rewriter/generator all go through PromptManager (no direct file loads)

4. **LLM client** — `get_llm(temperature=..., max_tokens=..., model=...)` with cache; parameters come from prompt metadata

5. **CLI** — `python -m evaluation.prompt_experiments` (`--list`, `--activate key:version`, run experiments)

6. **Tests** — `tests/test_prompt_registry.py`; SQL/RAG/planner tests updated for registry path

7. **Docs** — README Prompt Registry section + env vars; this session log

---

## How to use

```bash
export PYTHONPATH=.
python -m evaluation.prompt_experiments --list
export PROMPT_EXPERIMENT=planner.planner_prompt:v2
python -m evaluation.prompt_experiments --activate planner.planner_prompt:v2 --persist
python -m evaluation.prompt_experiments --mode mock
pytest tests/test_prompt_registry.py -q
```

Debug: `PROMPT_DEBUG=true` or `DEBUG=true`.

---

## Important Decisions

- Components never open prompt files; only PromptManager/Registry.
- Temperature / max_tokens / model live in prompt frontmatter, not call sites.
- Active version switches via `registry.yaml` or `PROMPT_EXPERIMENT` without code changes.
- Experiments reuse the evaluation framework; priorities configurable per experiment.

---

## Notes for Future Agents

- Add new prompts as `prompts/<family>/<name>_vN.md` + update that family's `registry.yaml`.
- Prefer `manager.invoke(...)` or `build_messages` + `manager.llm_for(selection)`.
- Do not reintroduce hardcoded system prompts in tools.
- Expand frontmatter `includes:` for shared rules instead of duplicating policy text.

---

*End of session 2026-07-22 (Prompt Registry).*

---

# Session — 2026-07-23 (Multi-Agent Architecture + Agent Registry)

## Session Information

| Field | Value |
|-------|--------|
| **Date** | 2026-07-23 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Refactor execution into Supervisor + Agent Registry with specialized agents; Planner routes by capability; Supervisor has no business logic |
| **Test status** | **`381 passed`** (full suite) |

---

## Work Completed

1. **`agents/` package**
   - `base.py` — `BaseAgent`, `AgentTask`, `AgentResponse`, `AgentHealth`, `AgentCapability`, `TOOL_TO_CAPABILITY`
   - `registry/agent_registry.py` — register / discover / select by capability / health
   - `supervisor/supervisor_agent.py` — capability dispatch, parallel waves, merge
   - Specialized: RAG, SQL, Analytics, General
   - Support: Memory, Reflection, Recommendation
   - Thin adapters over existing `plan_executor` tool handlers (`agents/_bridge.py`)

2. **Planner** — `ExecutionPlan.required_capabilities` populated in `_finalize_plan`

3. **PlanExecutor** — when `MULTI_AGENT_ENABLED` + default tool registry → Supervisor; else ToolOrchestrator / sequential

4. **Config** — `MULTI_AGENT_*` flags; tests disable multi-agent by default in `conftest.py`

5. **Tests** — `tests/test_multi_agent.py`

---

## How to use

```bash
export MULTI_AGENT_ENABLED=true
export MULTI_AGENT_DEBUG=true
pytest tests/test_multi_agent.py -q
```

Add a new agent: subclass `BaseAgent`, declare capabilities, `registry.register(agent)` — do not edit Supervisor.

---

## Important Decisions

- Supervisor never hardcodes agent names; only capability → Registry lookup.
- Agents wrap existing tools/services (no duplicated SQL/RAG business logic).
- Parallel execution for independent plan steps (dependency waves).
- Unhealthy agents skipped when healthier alternatives exist.

---

## Notes for Future Agents

- Future: Knowledge Graph / Forecast / Search / Vision / API agents — register with new capabilities only.
- Keep `MULTI_AGENT_ENABLED=false` in unit tests that assert ToolOrchestrator / custom tool registries.
- Do not put domain logic in Supervisor; put it in specialized agents or `tools/`.

---

*End of session 2026-07-23 (Multi-Agent Architecture).*


---

# Session 2026-07-26 — Attachment column recognition & file QA hardening

## Session Information

| Item | Value |
|------|--------|
| **Date** | 2026-07-26 |
| **Branch** | `new_feature_1` |
| **Tip commit** | `8bf72fe` — *Keep uploaded-file analysis on real columns, including Chinese headers.* |
| **Overall objective** | When a user attaches a data file, the agent must read it, recognize real column names (including Chinese), and use those columns for follow-up analysis—without falling back to the ops SQLite schema or overwriting file answers with SQL QA retries. |

---

## Work Completed

### Features implemented

1. **File-column recognition in ADA**
   - Match question tokens to DataFrame headers (exact, whitespace/newline-normalized, Chinese substrings).
   - Aggregation / ranking / visualize prefer the named dimension (e.g. `发件人详细地址`).
   - When the user asks “how many packages for each …” and no `package_count` column exists, use **row counts** per group (waybill-style files).

2. **Attachment-preferring intent routing**
   - File session / stored attachments + column / for-each / distribution language → `ATTACHMENT_ANALYSIS` or `ATTACHMENT_VISUALIZATION`.
   - Clear ops-DB asks (e.g. “Rank all hubs”) still detach to `SQL_ANALYTICS`.

3. **Substantive conversation repair**
   - “No, for \<column\> …” is treated as a full replacement ask, not a patch of the previous “inspect this file” into “inspect this file by packages”.

4. **Analyzer prefers original wording** for aggregation / column asks so repaired summary prompts cannot erase named columns.

### Bugs fixed

1. **“Inspect the file” → “No matching operational records” / `SELECT 'UNKNOWN'`**
   - Reflection/QA treated dataframe preview rows in `sql_rows` as warehouse evidence → `retry_sql` → empty ops answer.
   - Fixed by skipping QA/reflection for `ATTACHMENT_*` / `file_*` capabilities (`_should_run_qa`, `is_file_capability`).

2. **Follow-up on Chinese column → SQL `address_id`**
   - Router defaulted to SQL (packages/addresses language) or detached after non-followup long questions.
   - Fixed by file-focused routing + column-analysis phrases.

3. **Follow-up stayed on attachment but returned Executive Summary**
   - Repair cue `"no"` rewrote question to `inspect this file by packages`.
   - Fixed in `ConversationResolver._resolve_repair` + analyzer question preference.

### Backend / agent / Docker improvements

- `docker-compose.yml`: uvicorn `--reload`; mount `agents/` + `prompts/`.
- `Dockerfile` CMD includes `--reload` for consistency.
- Charts accept `preferred_dimension` / `preferred_metric`; skip time chart when a preferred dimension is set.
- Word-boundary repair phrase matching for short tokens like `"no"` in `tools/memory/state.py` (aligned with resolver).

### Tests added/updated

- Router: attachment column follow-up stays on file.
- ADA: Chinese address column aggregation counts rows.
- Conversation resolver: “no, for 发件人详细地址 …” keeps column ask.
- QA: file capabilities do not trigger SQL retry paths (existing suite extended earlier in the lineage).

### Documentation

- Rewrote `README.md` for current branch tip (`8bf72fe`).
- This session entry appended to `DEVELOPMENT_LOG.md`.

### Not in this session (already on branch from prior commits)

- Prompt Registry (`prompts/`, `core/prompt_*.py`) — tip `74924e2`
- Multi-agent Supervisor + Registry (`agents/`) — tip `74924e2`

---

## Files Created

None. All work extended existing modules.

*(Runtime-only e2e fixture `data/uploads/_e2e_cn_address.xlsx` may exist locally; do not treat as a product source file.)*

---

## Files Modified

| File | Why |
|------|-----|
| `core/intent_router.py` | Prefer attachment for file-column / for-each; keep ops-DB detach for “rank all hubs”; `_mentions_file_column_analysis` |
| `core/agent.py` | `_should_run_qa` skips attachment/file answers; align classification with attachment routes |
| `core/quality_assurance.py` | `is_file_capability`; never `should_retry_sql` for file modes |
| `core/reflection.py` | Respect file-capability / no SQL retry on ADA |
| `tools/files/data_analysis.py` | Column matching; row-count aggregation; dimension/metric inference for Chinese headers |
| `tools/files/analysis_intent.py` | AGGREGATION for “for each”, “distribution”, “how many packages…”, column+count |
| `tools/files/analyzer.py` | `_prefer_file_analysis_question` |
| `tools/files/charts.py` | Preferred dimension/metric for bar charts |
| `tools/conversation/resolver.py` | Substantive repair replacement; word-boundary `_is_repair` |
| `tools/memory/state.py` | Word-boundary `_is_repair` for short phrases |
| `docker-compose.yml` | `--reload`; mount `agents/`, `prompts/` |
| `Dockerfile` | CMD `--reload` |
| `tests/test_intent_router.py` | Column follow-up stays on attachment |
| `tests/test_ada_attachment_analysis.py` | Chinese column aggregation |
| `tests/test_conversation_resolver.py` | “No, for column…” keeps ask |
| `tests/test_quality_assurance.py` | File capability QA behavior |
| `README.md` | Full current-state rewrite (2026-07-26) |
| `DEVELOPMENT_LOG.md` | This session |

---

## Problems Encountered

### Problem 1 — Inspect file answered with empty ops SQL

| | |
|--|--|
| **Problem** | Upload + “inspect this file” returned “No matching operational records” / `SELECT 'UNKNOWN'`. |
| **Root Cause** | Router correctly chose `ATTACHMENT_ANALYSIS` and ADA produced a summary, but Reflection/QA treated preview rows as SQL evidence and forced `retry_sql`, overwriting the file answer. |
| **Solution** | Skip QA/reflection for attachment routes and `file_*` capabilities; never set SQL retry for file answers. |
| **Lessons Learned** | File ADA and warehouse SQL must not share the same evidence validators without a capability gate. |

### Problem 2 — Column ask used `address_id` from SQLite

| | |
|--|--|
| **Problem** | User asked for package counts by `发件人详细地址`; agent grouped by `address_id` via `pickups`/`addresses`. |
| **Root Cause** | Long follow-up was not treated as attachment-focused; default/SQL path detached the file session. ADA also lacked Chinese column matching and “for each” aggregation intent. |
| **Solution** | File-focused router rules + column match + AGGREGATION intent + row counts. |
| **Lessons Learned** | “packages” + “addresses” is not enough to imply the ops warehouse when an upload is active and the user names a file column. |

### Problem 3 — Attachment route correct but Executive Summary returned

| | |
|--|--|
| **Problem** | After inspect, “no, for 发件人详细地址 column…” stayed on `ATTACHMENT_ANALYSIS` but answer was still a file summary. |
| **Root Cause** | `ConversationResolver` treated leading `"no"` as a repair and rewrote to `inspect this file by packages`; ADA intent became `EXECUTIVE_SUMMARY`. |
| **Solution** | Substantive replacement keeps the new ask; analyzer prefers original wording for aggregation/column intents. |
| **Lessons Learned** | Repair must distinguish short corrections (“no, hubs”) from full replacement asks that include new analysis language. |

---

## Important Decisions

1. **File answers never enter the SQL QA/reflection loop** — protects ADA from `SELECT 'UNKNOWN'` overwrite.
2. **Ops-DB detach still wins for explicit warehouse questions** (“rank all hubs”) even during an attachment session.
3. **Waybill files without `package_count` use COUNT(\*)** for “how many packages for each \<column\>”.
4. **Prefer user’s literal question for ADA** when it is more specific than a repaired summary prompt.
5. **Docker mounts `agents/` + `prompts/`** so Prompt Registry and multi-agent code reload with the API.
6. **Extend modules; do not recreate** Prompt Registry / multi-agent / clarification / evaluation stacks.

---

## Testing

### Unit / targeted

```bash
pytest tests/test_intent_router.py tests/test_ada_attachment_analysis.py tests/test_conversation_resolver.py -q
# Result: passed (25+ including new cases)
```

### Collection

```bash
pytest --collect-only -q
# Result: 385 tests collected
```

### Live API (Docker `:8000`)

1. `POST /attachments` with Chinese-column xlsx + `session_id`
2. `POST /ask` inspect with `attachments: [id]` → Executive Summary listing `发件人详细地址`
3. `POST /ask` column follow-up → aggregation rows with address text + `package_count` (e.g. 3 / 1 / 1)
4. Assert route `ATTACHMENT_ANALYSIS`, empty/non-ops SQL, no `address_id`

**Result:** E2E PASS

### Remaining issues

- `last_attachment_file_types` sometimes stays `[]` in conversation state after ADA (routing still works via other signals; worth hardening).
- High-cardinality address charts may still need smarter top-N labeling.
- Full `pytest tests/ -q` not re-run end-to-end in this session after the final doc update (targeted suites + e2e verified).

---

## Next Recommended Tasks

1. **Persist attachment metadata in state** — always set `last_attachment_file_types`, filenames, and active sheet after successful ADA so routing/context is robust when `attachment_active` flickers.
2. **Cache invalidation on re-upload** — clear stale DataFrames in long-lived Docker sessions when the same session uploads a new file.
3. **Chart UX for Chinese addresses** — truncate/label top-N groups; avoid pie/status charts stealing focus when preferred dimension is set.
4. **Optional: pass file schema columns into IntentRouter** — if the question mentions an exact stored column name, force attachment even without “column” English keyword.
5. **CI** — GitHub Action running `pytest` on push to `new_feature_1` / `main`.
6. **Deprecation cleanup** — replace `datetime.utcnow()` in memory modules.
7. **Do not** recreate Prompt Registry, multi-agent Supervisor, or Clarification Manager — extend only.

---

## Notes

- API upload endpoint is **`POST /attachments`**; ask body field is **`attachments`** (list of IDs).
- Attachment path bypasses Planner when route is `ATTACHMENT_*` (by design in `GOFOAgent`).
- Screenshot symptom that started this work: agent answered with `address_id` volumes from ops SQL while acknowledging `发件人详细地址` in prose — root causes were routing + repair + missing column match, not missing Excel parse.
- Commit `8bf72fe` pushed to `origin/new_feature_1`.
- For a brand-new Cursor agent: read `README.md` (current state), then this session, then inspect `core/intent_router.py`, `tools/files/data_analysis.py`, `tools/conversation/resolver.py`, and `core/agent.py` (`_should_run_qa`) before changing attachment behavior.

---

*End of session 2026-07-26 (Attachment column recognition & file QA hardening).*

---

# Session 2026-07-26 — Persist attachment metadata across turns

## Session Information

| Item | Value |
|------|--------|
| **Date** | 2026-07-26 |
| **Branch** | `new_feature_1` |
| **Overall objective** | Persist attachment metadata in `ConversationState` after successful ADA so IntentRouter can continue file analysis when `attachment_active` becomes false between turns |
| **Tests** | `tests/test_attachment_metadata_persistence.py` + router/state suites green |

---

## Work Completed

### Why this was needed

`attachment_active` is a per-turn bind flag. After a successful upload + ADA turn, follow-ups often arrive with no new upload and `attachment_active=False`. Weak SQL-term matching (e.g. “delayed” + “pickups”) could steal the route to SQL/chat even though the user was still talking about the uploaded file. Separately, `GOFOAgent` enriched `file_context_summary` **after** `ConversationState.update()`, so `last_attachment_file_types` often never got populated.

### Features implemented

1. **`ConversationState` attachment metadata**
   - Fields: `last_attachment_file_types`, `last_attachment_filenames`, `last_active_sheet`, `attachment_context_timestamp`
   - Persist only non-empty values; empty ADA summaries never wipe prior metadata
   - `clear_attachment_context()` for explicit conversation reset
   - Debug log: `[Memory] Persisted attachment metadata …`

2. **Agent wiring**
   - Enrich `attachment_ids` / `attachment_filenames` / `file_context_summary` (incl. `active_sheet`) **before** `state.update`
   - `GOFOAgent.clear_attachment_context()` clears AttachmentMemory + state metadata
   - Detach still clears only the bind flag — metadata remains

3. **IntentRouter fallback**
   - File session = new upload / active bind / (stored or persisted metadata + prior `ATTACHMENT_*` route)
   - Inside a file session, only **explicit** ops-DB asks detach (rank-all / database phrases), not weak term counts
   - Debug log: `[Router] Using persisted attachment context …` when reusing metadata with `attachment_active=False`

### Tests

- Upload Excel → follow-up without upload → ADA
- Upload PDF → follow-up → ADA
- `attachment_active=False` + metadata → ADA
- Empty metadata does not overwrite
- New upload replaces metadata
- Worksheet switch updates `last_active_sheet`
- Conversation reset clears metadata
- “Rank all hubs” still detaches to SQL

### Files modified

| File | Why |
|------|-----|
| `tools/memory/state.py` | Persist / clear attachment metadata; snapshot fields |
| `core/agent.py` | Enrich response before state update; `clear_attachment_context`; active sheet helper |
| `core/intent_router.py` | Persisted-metadata file session; explicit ops detach; reuse logging |
| `tests/test_attachment_metadata_persistence.py` | New coverage |
| `README.md` | Current-state docs for persistence |
| `DEVELOPMENT_LOG.md` | This session |

### Important decisions

1. Metadata outlives `attachment_active` — detach ≠ clear.
2. Replace only on successful non-empty ADA/upload (or sheet change); never with empties.
3. Explicit conversation reset is the only bulk clear path.
4. Explicit warehouse asks still win over file session.

### Remaining / next

- Stale DataFrame cache invalidation on re-upload in long-lived Docker sessions
- Chart UX for high-cardinality Chinese address columns
- Broader PlanExecutor coverage for Follow_Up / Upload_File

---

*End of session 2026-07-26 (Persist attachment metadata across turns).*

---

# Session 2026-07-26 — Fix ADA follow-ups stuck on Executive Summary

## Session Information

| Item | Value |
|------|--------|
| **Date** | 2026-07-26 |
| **Branch** | `new_feature_1` |
| **Overall objective** | After a successful file summary, follow-ups like “which address has most packages” and “Create a chart … addresses … packages volume” were repeating the Executive Summary instead of ranking/visualizing |
| **Tests** | `tests/test_ada_attachment_analysis.py` — **13 passed** |

---

## Root Cause

`detect_analysis_intent()` returned `GENERAL` for those prompts (no “highest/for each/show chart” phrases). `analyze_dataframe` maps `GENERAL` → `_executive_summary`, so every follow-up looked identical to “summarize this file”.

Separately, `_ranking` required a real `package_count` column; waybill files without that metric fell back to Executive Summary even when intent was correct.

## Fix

1. **`analysis_intent.py`**
   - Treat bare `chart` / “create a chart” as `VISUALIZE`
   - Treat “which address has most packages” / address+package volume language as `RANKING` or `AGGREGATION`

2. **`data_analysis.py`**
   - Ranking uses row-count when no `package_count` (same as aggregation)
   - Visualize result carries preferred dimension/metric

3. **Tests** for both user prompts against Chinese address columns

## Files Modified

| File | Why |
|------|-----|
| `tools/files/analysis_intent.py` | Broader ranking / visualize / address+package detection |
| `tools/files/data_analysis.py` | Row-count ranking; visualize metadata |
| `tests/test_ada_attachment_analysis.py` | Regression for the screenshot follow-ups |
| `DEVELOPMENT_LOG.md` | This session |

---

*End of session 2026-07-26 (Fix ADA follow-ups stuck on Executive Summary).*

---

# Session 2026-07-26 — Distinguish meta / SOP / uploaded-file analytics routing

## Session Information

| Item | Value |
|------|--------|
| **Date** | 2026-07-26 (docs tip refreshed 2026-07-27) |
| **Branch** | `new_feature_1` |
| **Tip commit** | `f8d0be9` — *Fix attachment follow-up routing so ADA, SOP, and meta asks stay distinct.* |
| **Overall objective** | Fix wrong answers when users mix meta file questions, SOP glossary asks, and uploaded-file analytics / follow-ups |
| **Tests** | Router + ADA + meta dispatch suites — **43 passed** (full collect **403**) |

---

## Bugs (from screenshots)

1. “Which file are you reading now / previously?” → Executive Summary (ADA)
2. “What is CBT?” (SOP) → API timeout (heavy planner path)
3. “In the data file I just uploaded… address with most packages” after SOP → ops SQL `address_id` (not the upload)
4. “Which address is it, give me the address name” → SOP / fallback instead of ADA follow-up

## Fixes

1. **IntentRouter**
   - Meta attachment questions → `GENERAL_CHAT` / handler `Attachment Context` (no ADA summary)
   - Explicit upload references + address/package analytics reactivate ADA even after SOP/SQL
   - Address-name follow-ups return to ADA when stored/persisted attachments exist
   - Pure SOP glossary still detaches from the file session

2. **RouteDispatcher**
   - Answers meta file questions from `last_attachment_filenames` / `previous_attachment_filenames`

3. **ADA**
   - `FILE_CONTEXT` intent; LOOKUP for address-name follow-ups using `last_entity`

4. **ConversationState**
   - Tracks `previous_attachment_filenames` when a new upload replaces the current one

5. **GOFOAgent / Streamlit**
   - Short “what is X” SOP asks skip planner/multi-agent path
   - `/ask` client timeout raised to 300s
   - Dispatcher receives ConversationState snapshot (filenames) merged into conversation_state

## Files Modified

| File | Why |
|------|-----|
| `core/intent_router.py` | Meta / upload-reactivate / entity-followup routing |
| `core/route_dispatcher.py` | Meta attachment answers |
| `core/agent.py` | Simple SOP glossary bypass; pass state snapshot |
| `tools/files/analysis_intent.py` | `FILE_CONTEXT` + address LOOKUP |
| `tools/files/data_analysis.py` | Meta + address-name lookup handlers |
| `tools/memory/state.py` | `previous_attachment_filenames` |
| `frontend/app.py` | Longer ask timeout |
| `tests/test_intent_router.py` | New routing cases |
| `tests/test_attachment_meta_dispatch.py` | Meta answer regression |
| `tests/test_ada_attachment_analysis.py` | FILE_CONTEXT / LOOKUP |
| `README.md` / `DEVELOPMENT_LOG.md` | Docs |

### Documentation note (2026-07-27)

`README.md` was refreshed so tip `f8d0be9`, test count **403**, `previous_attachment_filenames`, meta/SOP/file routing table, SOP glossary planner bypass, and Streamlit `/ask` timeout **300s** match the code. This DEVELOPMENT_LOG session already covered the engineering work; no feature code changed in the docs refresh.

---

*End of session 2026-07-26 (Distinguish meta / SOP / uploaded-file analytics routing).*

