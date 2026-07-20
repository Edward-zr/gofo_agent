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
