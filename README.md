# GOFO Operations Intelligence Agent

This README is the permanent source of truth for the **current** state of the project.  
Historical engineering sessions live in [`DEVELOPMENT_LOG.md`](./DEVELOPMENT_LOG.md).

**Future Cursor agents: read this file first, then inspect the repository, then continue from the architecture and roadmap below. Do not recreate completed modules.**

---

# Project Overview

## Purpose

`gofo_agent` is an AI **operations intelligence copilot** for GOFO logistics teams. It answers SOP/policy questions, analyzes pickup operations from a SQLite warehouse, investigates root causes, detects anomalies, supports multi-turn operational conversation, and analyzes uploaded files (Excel/CSV/PDF/images) with ChatGPT-style Advanced Data Analysis (ADA) behavior—including **real file column recognition** (English and Chinese headers).

## Business Goal

Give operations users one chat surface for:

- SOP / policy questions (hybrid RAG over Chroma)
- Live operational metrics (SQL over SQLite)
- Multi-turn follow-ups and corrections (“Actually I mean lowest driver”)
- Uploaded logistics exports (preview + analyze + Python-generated charts + group-by named columns)
- Multi-step analytical workflows with quality critique and bounded retries

## Current Development Stage

**Active feature branch with a production-style local/Docker service and a multi-stage agent pipeline.**

| Item | Value |
|------|--------|
| **Branch tip** | `8bf72fe` — *Keep uploaded-file analysis on real columns, including Chinese headers.* |
| **Prior tip** | `74924e2` — Prompt Registry + Multi-Agent Supervisor |
| **Tests** | **385** collected (`pytest --collect-only`) |
| **Runtime** | FastAPI `:8000` + Streamlit `:8501` via Docker Compose (`--reload` + source mounts) |

Completed control plane: Intent Classifier → Data Source Selection → Planner → Clarification → Supervisor/Orchestrator → Generator → Reflection/QA (skipped for file ADA). Latest hardening: attachment session stays on uploaded columns; conversation repair no longer rewrites file-column asks into summary prompts; **attachment metadata persists across turns** so ADA follow-ups keep working when `attachment_active` flickers false.

---

# Branch Information

| Item | Value |
|------|--------|
| **Current branch** | `new_feature_1` |
| **Tracks** | `origin/new_feature_1` |
| **Base** | Evolved from `cursor-memory-version` / early `main` checkpoint history |
| **Purpose** | Ship conversational GOFO intelligence with durable docs, multi-step planning, multi-agent dispatch, Prompt Registry, and reliable uploaded-file ADA |

## Major Differences from `main`

Relative to the early `main` checkpoint, this line of development adds:

1. **Centralized intent routing** (`IntentRouter` + `RouteDispatcher`) with attachment activate/detach
2. **GPT + heuristic Intent Classifier** (`core/intent_classifier.py`)
3. **Data Source Selection** (`core/data_source_selector.py`)
4. **Clarification Manager** (`core/clarification_manager.py`) — ask before tools; resume plan after answer
5. **Multi-step Planner** (`core/planner.py`) — `ExecutionPlan` / `required_capabilities` (never answers, never runs tools)
6. **Knowledge Graph** (`tools/knowledge_graph/`)
7. **Tool Orchestrator** + **PlanExecutor** — sequential/parallel waves; AgentState
8. **Multi-Agent Architecture** — Supervisor + Agent Registry (capability → agent)
9. **Enterprise Prompt Registry** — versioned assets under `prompts/`
10. **Reflection + Quality Assurance** — bounded retries; **file ADA skips SQL QA**
11. **End-to-end multimodal attachments** — upload, preview, ADA, matplotlib charts
12. **File-column recognition** — match question text to real DataFrame headers (incl. Chinese); row-count aggregation when no `package_count`
13. **Persisted attachment metadata** — `last_attachment_file_types` / filenames / active sheet survive across turns; empty ADA responses never erase prior values
14. **Hybrid RAG** — Chroma dense + BM25 + RRF + cross-encoder rerank + confidence engine
15. **Excel robustness** — shared `excel_reader` (no primary openpyxl `read_only`)
16. **Docker source mounts + CJK fonts** — live iteration; Chinese chart labels
17. **Documentation split** — README = current state; DEVELOPMENT_LOG = append-only history

---

# System Architecture

## High-Level Request Flow

```text
User (Streamlit / CLI)
  → FastAPI (session_id → SessionManager → GOFOAgent)
  → optional POST /attachments → AttachmentService.upload
  → IntentClassifier.classify(question, ConversationMemory)
  → IntentRouter.route(...)          # attachment activate/detach + WAIT_FOR_UPLOAD
                                     # file-column / for-each asks stay on ATTACHMENT_*
  → (if planner path) DataSourceSelector → Planner → ClarificationManager
       → Supervisor / Agent Registry   # MULTI_AGENT_ENABLED (default true)
       → else ToolOrchestrator / PlanExecutor
  → (if attachment path) RouteDispatcher → analyze_attachments / ADA
       → detect_analysis_intent → analyze_dataframe (real columns)
       → charts.build_charts (matplotlib PNG base64)
  → Reflection/QA ONLY if _should_run_qa(...)  # SKIPPED for ATTACHMENT_* / file_*
  → ConversationMemory + ConversationState + AttachmentMemory update
  → AskResponse (answer, sql, data, kpi, charts, sources, plan, reflection, agent_state)
  → Streamlit renders text + KPI + chart PNGs
```

**Separation of duties (do not collapse these):**

| Role | Module | Responsibility |
|------|--------|----------------|
| Classify | `IntentClassifier` | *What* the user wants |
| Select sources | `DataSourceSelector` | *Which* data sources (min set) |
| Plan | `Planner` | *What steps* (never executes); emits capabilities |
| Clarify | `ClarificationManager` | *Ask* for missing business params before tools |
| Supervise | `SupervisorAgent` + `AgentRegistry` | *Which agents* by capability; dispatch/merge |
| Orchestrate | `ToolOrchestrator` | Legacy *how* when `MULTI_AGENT_ENABLED=false` |
| Critique | `ReflectionAgent` + QA | Evaluate draft SQL/RAG answers (**not** file ADA) |
| Route (session) | `IntentRouter` / `RouteDispatcher` | Attachment session + single-route fallback |
| ADA | `tools/files/*` | Parse-once DataFrame; per-prompt analysis |

There is **no LangGraph** dependency. Stages are modular so a future graph could wrap the same modules.

## Frontend

| Component | Path | Role |
|-----------|------|------|
| Primary UI | `frontend/app.py` | Streamlit chat; composer; KPIs; matplotlib chart images |
| Attachment UI | `frontend/attachment_ui.py` | Upload “+”, history chips, preview buttons (stable keys) |
| Preview | `frontend/attachment_preview.py` | Modal preview for CSV/Excel/PDF/DOCX/text/image |
| Legacy UI | `ui/app.py` | Older Streamlit client; not primary |

Frontend talks only to FastAPI (`API_BASE_URL`, default `http://localhost:8000`).

## Backend / API

| Component | Path | Role |
|-----------|------|------|
| FastAPI app | `api/server.py` | `GET /health`, `GET /`, `POST /attachments`, `POST /ask` |
| Schemas | `api/schemas.py` | `AskRequest` (`question`, `session_id`, `attachments`), `AskResponse` |
| Sessions | `core/session.py` | `session_id` → long-lived `GOFOAgent` |

**Important:** Session agents share the process-wide `AttachmentService` so uploads and asks use the same in-memory index. Ask body field is **`attachments`** (list of IDs), not `attachment_ids`.

## AI Agent

| Component | Path | Role |
|-----------|------|------|
| Agent facade | `core/agent.py` | `GOFOAgent.ask()` — full pipeline; `_should_run_qa` skips file routes |
| Intent classifier | `core/intent_classifier.py` | Primary intent + tool flags + confidence |
| Data source selector | `core/data_source_selector.py` | Minimum SQL/KG/RAG/Memory/Python sources |
| Planner | `core/planner.py` | Multi-step `ExecutionPlan` / `plan_retry` |
| Plan executor | `core/plan_executor.py` | Supervisor when multi-agent on; else ToolOrchestrator |
| Tool orchestrator | `core/tool_orchestrator.py` | Parallel waves, deps, AgentState |
| Quality assurance | `core/quality_assurance.py` | Evidence / reasoning / completeness; `is_file_capability()` |
| Reflection | `core/reflection.py` | Self-critique API for Planner retries |
| Intent router | `core/intent_router.py` | Attachment activate/detach; file-column preference; persisted-metadata fallback |
| Dispatcher | `core/route_dispatcher.py` | ATTACHMENT_* → `analyze_attachments` |
| Prompt Registry | `core/prompt_*.py` | Versioned prompt load / render / invoke |
| Models | `core/models.py` | `QueryResponse` (+ plan, QA, reflection, agent_state) |

## Prompt Registry

Every LLM interaction loads prompts through the **Prompt Registry**. Prompts are versioned assets with YAML frontmatter — not ad-hoc strings in Python.

```text
Component → PromptManager.get / render / invoke
         → PromptRegistry → PromptRenderer → LLM (params from frontmatter)
```

Asset root: `prompts/` (`router/`, `planner/`, `sql/`, `rag/`, `python/`, `recommendation/`, `reflection/`, `clarification/`, `shared/`, `experiments.yaml`).

```bash
export PROMPT_EXPERIMENT=planner.planner_prompt:v2
python -m evaluation.prompt_experiments --list
python -m evaluation.prompt_experiments --mode mock
```

**Rule:** never `open()` prompt files from tools; never hardcode temperature/max_tokens at call sites.

## Multi-Agent Architecture

Planner emits **capabilities**, not agent names. Supervisor → Registry → specialized agents → merge.

| Agent | Capabilities |
|-------|----------------|
| RAG | `sop_qa`, `document_retrieval`, `document_summarization` |
| SQL | `sql_planning`, `sql_generation`, `database_query` |
| Analytics | `statistics`, `charts`, `data_analysis` |
| General | `general_knowledge`, `brainstorming`, `conversation` |
| Memory | `conversation_memory`, `context_tracking`, `slot_reuse` |
| Reflection | `answer_review`, `consistency_check`, `hallucination_detection` |
| Recommendation | `suggested_analyses`, `operational_recommendations`, `next_steps` |

**Layout:** `agents/supervisor/`, `agents/registry/`, `agents/{rag,sql,analytics,general,memory,reflection,recommendation}/`

Add agents via `BaseAgent` + `registry.register` — **do not edit the Supervisor**.

Flags: `MULTI_AGENT_ENABLED` (default true), `MULTI_AGENT_PARALLEL`, `MULTI_AGENT_DEBUG`, `MULTI_AGENT_REFLECTION`.

## SQL Analytics Pipeline

```text
Question → Schema Retriever → SQL Generator → Validator → Executor (SQLite) → Summarizer
```

| Component | Path |
|-----------|------|
| Schema Registry | `tools/sql/schema_registry.py` |
| Schema Retriever | `tools/sql/schema_retriever.py` |
| Planner/Generator | `tools/sql/planner.py` |
| Validator | `tools/sql/validator.py` |
| Service | `tools/sql/service.py` |

## RAG Pipeline

```text
docs/ → ingest.py → Chroma + bm25_corpus.json
query → hybrid retrieve (dense + BM25 → RRF → rerank)
      → Retrieval Confidence Engine → Generator
```

| Component | Path |
|-----------|------|
| Hybrid retriever | `tools/rag/hybrid.py`, `retriever.py` |
| Confidence | `tools/rag/confidence.py` |
| Service / Generator | `tools/rag/service.py`, `generator.py` |

## Vector Database

- **ChromaDB** under `chroma_db/`
- Collection: `gofo_sop` (`COLLECTION_NAME`)
- Embeddings: OpenAI `text-embedding-3-small` (default)
- Reranker: `BAAI/bge-reranker-base` via `sentence-transformers`

## Databases

| DB | Path | Purpose |
|----|------|---------|
| Analytics SQLite | `data/gofo_demo.db` | Pickups, drivers, customers, hubs, addresses |
| Memory SQLite | `data/memory.db` | Conversations, findings, patterns |
| Knowledge Graph | `tools/knowledge_graph/` | Driver→Hub→Region, Manager→Hub, SOP ownership |
| Upload store | `data/uploads/` | Attachment bytes + `attachments_index.json` |

## Attachment / ADA Pipeline (critical)

```text
POST /attachments → AttachmentService.upload
GOFOAgent.ask(attachments=[id])
  → get_many_processed → AttachmentMemory.register_contexts
  → IntentRouter → ATTACHMENT_ANALYSIS | ATTACHMENT_VISUALIZATION
  → RouteDispatcher._dispatch_attachment
  → analyze_attachments
       → prefer original question wording (not repaired summary)
       → detect_analysis_intent (AGGREGATION / SUMMARY / …)
       → analyze_dataframe
            → _match_columns_in_question (exact / compact / Chinese)
            → groupby real column; COUNT(*) when no package_count
       → charts.build_charts (preferred_dimension / preferred_metric)
  → SKIP Reflection/QA (_should_run_qa = False)
  → return answer + sql_rows (preview / aggregation) + charts
```

### Attachment metadata persistence (across turns)

`ConversationState` keeps the last successful ADA context so follow-ups do not require re-upload when `attachment_active` becomes false:

| Field | Purpose |
|-------|---------|
| `last_attachment_file_types` | e.g. `["excel"]`, `["pdf"]` — router data-source hints |
| `last_attachment_filenames` | e.g. `["pickup_data.xlsx"]` — filename-aware routing later |
| `last_active_sheet` | Active worksheet for “summarize this sheet” |
| `attachment_context_timestamp` | UTC ISO time of last successful persist |

**Why:** `attachment_active` is a per-turn bind flag. Clearing it (no new upload, SQL/chat detach) must **not** erase file context. The IntentRouter treats prior attachment route + persisted metadata as a file session and prefers ADA unless the user makes an explicit ops-DB / SOP / chat ask.

**Replace when:** a new attachment is processed successfully, the active sheet changes, or ADA reports a new usable file. **Never** overwrite with empty lists/None from a sparse response.

**Clear when:** `GOFOAgent.clear_attachment_context()` / `ConversationState.clear_attachment_context()` (explicit conversation reset). Detach alone does not clear metadata.

**Rules that must not regress:**

1. Active attachment / file-column language stays on the file unless the user explicitly asks for live DB comparison or a clear ops-DB ask (“rank all hubs”).
2. Conversation repair must **not** rewrite “No, for \<column\>…” into “inspect this file by packages”.
3. Never treat dataframe preview rows as warehouse SQL evidence in QA.
4. Excel: never use openpyxl `read_only=True` as the primary reader (`tools/files/excel_reader.py`).
5. Empty `file_context_summary` values must not wipe persisted attachment metadata.

## Clarification Manager

Sits **after Planner, before tool execution**. Asks for missing metric / scope / time; resumes enriched question. Skip when memory already has slots.

## Specialized Python Analytics Tools

`tools/python/{transformation,statistics,visualization,recommendation}_tool.py` — LLM never calculates. Attachment ADA charts still use `tools/files/charts.py` directly.

## External Integrations

- **OpenAI** — chat (`LLM_MODEL`, default `gpt-4o-mini`) and embeddings
- **Hugging Face / sentence-transformers** — local reranker
- No Slack/Lark/Redis/Snowflake/LangGraph unless explicitly requested

---

# Repository Structure

```text
gofo_agent/
├── api/                 # FastAPI HTTP layer
├── agents/              # Multi-agent: Supervisor, Registry, specialized agents
├── core/                # Agent control plane + Prompt Registry
├── prompts/             # Versioned prompt assets (registry.yaml per family)
├── frontend/            # Primary Streamlit UI + attachment preview
├── tools/
│   ├── analysis/        # KPI, root cause, anomaly, recommendations
│   ├── analyzer/        # Business reasoner
│   ├── context/         # Conversation/file context builders
│   ├── conversation/    # Multi-turn resolver / repair (substantive "No, …" keep)
│   ├── files/           # Attachments, ADA, Excel reader, matplotlib charts
│   ├── llm/             # OpenAI client wrappers
│   ├── memory/          # Short-term, state, long-term SQLite memory
│   ├── orchestration/   # Semantic request analysis
│   ├── planner/         # Legacy capability planner + business Intent enum
│   ├── python/          # Transform / stats / viz / recommend
│   ├── knowledge_graph/ # Relationship lookups
│   ├── rag/             # Hybrid retrieval + confidence + generation
│   ├── sql/             # Schema registry/retriever, planner, executor, validator
│   ├── router.py
│   └── synthesizer.py
├── cli/                 # Interactive CLI REPL
├── evaluation/          # Benchmark + prompt experiments
├── tests/               # Pytest suite (~385 tests)
├── scripts/             # Demo DB + helpers
├── data/                # SQLite DBs + uploads (runtime; do not commit secrets)
├── chroma_db/           # Vector store persistence
├── docs/                # SOP source docs for ingest
├── assets/fonts/        # Optional bundled CJK fonts
├── config.py            # Central env-backed configuration
├── ingest.py            # RAG + BM25 ingest entrypoint
├── agent.py / query.py  # CLI / legacy entry helpers
├── docker-compose.yml   # gofo-api + gofo-ui (--reload, source mounts)
├── Dockerfile           # Python 3.11 + matplotlib + WenQuanYi CJK font
├── requirements.txt
├── .env.example
├── README.md            # ← current state (this file)
├── README_RUN.md        # Short local run cheat sheet
└── DEVELOPMENT_LOG.md   # Append-only engineering history
```

### Important files (responsibility)

| File | Responsibility |
|------|----------------|
| `core/agent.py` | Single facade; wires pipeline; skips QA for attachment routes |
| `core/intent_router.py` | Attachment session + file-column preference vs ops SQL detach; persisted-metadata fallback |
| `tools/memory/state.py` | ConversationState; attachment metadata persist / clear (never wipe with empties) |
| `core/intent_classifier.py` | Primary intent classification |
| `core/data_source_selector.py` | Minimum necessary data sources |
| `core/planner.py` | ExecutionPlan + capabilities (never executes) |
| `core/plan_executor.py` | Supervisor or ToolOrchestrator |
| `core/tool_orchestrator.py` | Parallel/deps/retries/AgentState |
| `core/quality_assurance.py` | Validators; `is_file_capability` |
| `core/reflection.py` | Critic API |
| `core/prompt_manager.py` / `prompt_registry.py` | Prompt load/render/invoke |
| `core/route_dispatcher.py` | ATTACHMENT_* → ADA |
| `core/clarification_manager.py` | Ask before tools |
| `tools/files/data_analysis.py` | Column match + aggregation / ranking / summary |
| `tools/files/analysis_intent.py` | Per-prompt ADA intent (AGGREGATION, VISUALIZE, …) |
| `tools/files/analyzer.py` | Prefer original wording for file analysis |
| `tools/files/charts.py` | Matplotlib PNG; preferred dimension/metric |
| `tools/files/excel_reader.py` | Robust Excel IO |
| `tools/conversation/resolver.py` | Follow-up resolve; substantive repair replacement |
| `tools/rag/hybrid.py` | Hybrid retrieval |
| `tools/sql/schema_retriever.py` | Relevant schema only |
| `api/server.py` | Thin HTTP; shared attachment service |
| `frontend/app.py` | Chat + chart rendering |
| `config.py` | All env defaults |
| `agents/supervisor/supervisor_agent.py` | Capability dispatch (no business logic) |
| `agents/registry/agent_registry.py` | Discover / health / select agents |

---

# Features

| Feature | Description | Files | Status |
|---------|-------------|-------|--------|
| SOP hybrid RAG | Dense + BM25 + RRF + rerank | `tools/rag/*`, `ingest.py` | **Complete** |
| Retrieval confidence | Multi-signal score + Planner fallbacks | `tools/rag/confidence.py` | **Complete** |
| SQL analytics | Schema retriever → generate → validate → execute | `tools/sql/*` | **Complete** |
| Specialized Python tools | Transform / Statistics / Visualization / Recommendation | `tools/python/*` | **Complete** |
| Business analysis | KPI, root cause, anomaly, recommendations | `tools/analysis/*` | **Complete** |
| Conversation memory | Short-term history + entity resolve | `tools/memory/*` | **Complete** |
| Conversation repair | Corrections; substantive “No, …” keep new ask | `tools/conversation/resolver.py` | **Complete** |
| Long-term memory | SQLite conversations/findings/patterns | `tools/memory/long_memory.py` | **Complete** |
| Intent router | Attachment session + file-column routing | `core/intent_router.py` | **Complete** |
| Intent classifier | Primary intents + confidence/tool flags | `core/intent_classifier.py` | **Complete** |
| Data source selection | Minimum SQL/KG/RAG/Memory/Python | `core/data_source_selector.py` | **Complete** |
| Clarification manager | Ask for missing business params | `core/clarification_manager.py` | **Complete** |
| Multi-step planner | ExecutionPlan + capabilities | `core/planner.py` | **Complete** |
| Knowledge graph | Driver/hub/region/manager/SOP | `tools/knowledge_graph/` | **Complete** |
| Tool orchestrator | Parallel/sequential multi-tool execution | `core/tool_orchestrator.py` | **Complete** |
| Multi-agent Supervisor | Capability routing via Registry | `agents/` | **Complete** |
| Prompt Registry | Versioned prompts + experiments | `core/prompt_*.py`, `prompts/` | **Complete** |
| Quality assurance | Evidence/reasoning/completeness | `core/quality_assurance.py` | **Complete** |
| Reflection critic | Bounded Planner retries | `core/reflection.py` | **Complete** |
| File QA skip | Attachment/file answers never enter SQL QA loop | `core/agent.py`, `quality_assurance.py` | **Complete** |
| File upload + ADA | Validate → process → analyze → charts | `tools/files/*` | **Complete** |
| File column recognition | Match question to real headers (incl. Chinese); COUNT per group | `data_analysis.py`, `analysis_intent.py` | **Complete** |
| ADA follow-up intents | Address ranking / chart asks do not fall back to Executive Summary | `analysis_intent.py`, `data_analysis.py` | **Complete** |
| Meta vs SOP vs file routing | “Which file…”, SOP glossary, and upload analytics stay on the correct path | `intent_router.py`, `route_dispatcher.py` | **Complete** |
| Attachment metadata persistence | File types / filenames / active sheet survive turns; empty values never wipe | `tools/memory/state.py`, `core/intent_router.py`, `core/agent.py` | **Complete** |
| Attachment preview | Sheet-aware modal preview | `frontend/attachment_preview.py` | **Complete** |
| Excel full-row read | Anti-truncation reader | `tools/files/excel_reader.py` | **Complete** |
| FastAPI + sessions | Persistent agent per `session_id` | `api/`, `core/session.py` | **Complete** |
| Streamlit UI | Chat composer, KPIs, charts | `frontend/app.py` | **Complete** |
| CLI | Local debug REPL | `cli/`, `query.py` | **Complete** |
| Docker Compose | API :8000 + UI :8501, `--reload`, mounts | `Dockerfile`, `docker-compose.yml` | **Complete** |
| Evaluation framework | Benchmark + scenarios + prompt experiments | `evaluation/` | **Complete** |

---

# Tech Stack

| Area | Choices |
|------|---------|
| **Language** | Python 3.11 (Docker); local 3.9–3.13 also used in dev |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **AI / LLM** | OpenAI Chat Completions (`gpt-4o-mini` default) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Reranker** | `sentence-transformers` + `BAAI/bge-reranker-base` |
| **RAG store** | ChromaDB |
| **Sparse retrieval** | `rank-bm25` |
| **Orchestration libs** | LangChain OpenAI wrappers (**not** LangGraph) |
| **Data** | pandas, openpyxl, pypdf, python-docx, Pillow |
| **Charts** | matplotlib (Agg → PNG base64) |
| **Analytics DB** | SQLite (`data/gofo_demo.db`) |
| **Memory DB** | SQLite (`data/memory.db`) |
| **Config** | python-dotenv, PyYAML, Pydantic |
| **Infra** | Docker, Docker Compose; source bind mounts; uvicorn `--reload` |
| **Fonts** | `fonts-wqy-zenhei` for Chinese chart labels |
| **Tests** | pytest |

---

# Installation

## Prerequisites

- Python 3.11+ recommended
- Docker Desktop (for Compose)
- OpenAI API key

## Local setup

```bash
cd gofo_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Ensure analytics DB exists:

```bash
python scripts/create_demo_db.py   # if needed
```

Ingest SOP docs (when `docs/` present):

```bash
python ingest.py
```

## Environment variables

Copy from `.env.example`. Critical keys:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for LLM + embeddings |
| `LLM_MODEL` | Default `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `SQLITE_DATABASE` | Analytics DB path |
| `MEMORY_DATABASE` | Long-term memory DB |
| `CHROMA_PERSIST_DIR` / `COLLECTION_NAME` | Vector store |
| `UPLOAD_DIR` / `MAX_UPLOAD_SIZE_MB` / `ALLOWED_UPLOAD_TYPES` | Attachments |
| `HYBRID_RETRIEVAL_ENABLED` | Hybrid RAG on/off |
| `DEBUG` | Classifier/planner/QA/orchestrator debug dumps |
| `INTENT_CLASSIFIER_ENABLED` | Primary IntentClassifier |
| `PLANNER_ENABLED` | Multi-step planner path |
| `CLARIFICATION_MANAGER_ENABLED` | Ask for missing params (default true) |
| `RETRIEVAL_CONFIDENCE_ENABLED` | Adaptive retrieval confidence |
| `DATA_SOURCE_SELECTION_ENABLED` | Minimum source selection |
| `SCHEMA_RETRIEVER_ENABLED` | Relevant schema before SQL |
| `QUALITY_ASSURANCE_ENABLED` | QA validators |
| `QA_MAX_RETRIES` / `QA_SCORE_THRESHOLD` / `QA_APPROVE_THRESHOLD` | QA thresholds |
| `REFLECTION_ENABLED` / `REFLECTION_USE_LLM` / `REFLECTION_MAX_RETRIES` | Critique loop |
| `TOOL_ORCHESTRATOR_*` | Plan execution |
| `PROMPT_REGISTRY_DIR` / `PROMPT_HOT_RELOAD` / `PROMPT_DEBUG` / `PROMPT_EXPERIMENT` | Prompt Registry |
| `MULTI_AGENT_ENABLED` / `MULTI_AGENT_PARALLEL` / `MULTI_AGENT_DEBUG` | Supervisor path |
| `API_BASE_URL` | Streamlit → API |
| `MPLCONFIGDIR` | Writable matplotlib cache (Compose: `/tmp/matplotlib`) |

**Never commit `.env` or real API keys.**

## Backend startup (local)

```bash
export PYTHONPATH=.
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Health / ask / upload:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/attachments -F "files=@report.xlsx" -F "session_id=demo"
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"inspect this file","session_id":"demo","attachments":["<id>"]}'
```

## Frontend startup (local)

```bash
export PYTHONPATH=.
export API_BASE_URL=http://localhost:8000
streamlit run frontend/app.py --server.port 8501
```

## Docker commands

```bash
docker compose up --build -d
docker compose logs -f gofo-api gofo-ui
docker compose up -d --force-recreate   # after compose volume changes
docker compose down
```

**Compose notes:**

- API runs `uvicorn … --reload`
- Bind-mounted: `api/`, `core/`, `tools/`, `agents/`, `prompts/`, `frontend/`, `config.py`, `data/`, `logs/`, `chroma_db/`
- Rebuild images after `requirements.txt` / Dockerfile changes
- Recreate containers after volume list changes

## CLI

```bash
export PYTHONPATH=.
python query.py
# or
python -m cli.repl
```

## Tests

```bash
export PYTHONPATH=.
pytest tests/ -q
```

Integration-heavy paths disable planner/QA/reflection/clarification execution by default in `tests/conftest.py`. Unit tests cover those modules directly. Multi-agent unit tests keep `MULTI_AGENT_ENABLED=false` when asserting ToolOrchestrator paths.

## Evaluation Framework

```bash
export PYTHONPATH=.
python -m evaluation                  # offline mock smoke
python -m evaluation --mode live      # real GOFOAgent
python -m evaluation.prompt_experiments --mode mock
```

Reports land under `evaluation/results/`.

---

# Current Architecture Decisions

These are intentional. Future agents should **not** reverse them without an explicit product request.

1. **Modular control plane, optional multi-agent** — one `GOFOAgent` facade; Supervisor/Registry when `MULTI_AGENT_ENABLED`; no LangGraph.
2. **Thin API/UI** — intelligence lives under `core/`, `tools/`, `agents/`.
3. **Classify → Select sources → Plan → Clarify → Supervise/Orchestrate → Critique** — do not collapse Planner into Executor or let critics write final answers.
4. **Planner never executes tools; Orchestrator/Supervisor never invent plans; DataSourceSelector never executes.**
5. **Minimum necessary sources** — combine SQL/KG/RAG/Python only when required.
6. **Reflection is the primary critic** for SQL/RAG; QA validators score underneath.
7. **Bounded retries (default 2)** — never infinite critique loops.
8. **File / attachment answers skip SQL QA & reflection** — dataframe rows in `sql_rows` are not warehouse evidence; never overwrite ADA with `SELECT 'UNKNOWN'`.
9. **Centralized IntentRouter owns attachment activate/detach** — uploads must not permanently hijack the session; ops-DB asks (“rank all hubs”) still detach.
10. **File-column / for-each language prefers attachment** while a file session or stored attachments exist.
11. **Persisted attachment metadata outlives `attachment_active`** — detach/flicker clears the bind flag only; IntentRouter reuses metadata for ADA follow-ups until explicit reset or a new file replaces it.
12. **Shared AttachmentService in API process** — session agents reuse the same upload index.
13. **Parse-once ADA** — store DataFrame; fresh `analyze_dataframe` per prompt; match real column names from the question.
14. **Substantive conversation repairs keep the new ask** — do not patch prior “inspect this file” when the user says “No, for \<column\>…”.
15. **Matplotlib server-side charts** — `image_base64` PNGs; UI uses `st.image`.
16. **Excel: no primary `read_only`** — use `tools/files/excel_reader.py`.
17. **Hybrid RAG behind `retrieve()`** — preserve public API.
18. **SQLite only for analytics + memory** — no Redis/Postgres unless requested.
19. **Docker source mounts + `--reload` for iteration** — recreate on volume changes; rebuild on dependency changes.
20. **Documentation split** — README = current; DEVELOPMENT_LOG = append-only history.
21. **SQL Schema Retriever** — never dump the full schema into the LLM prompt; validate against allowlist.
22. **Specialized Python tools** — never put calculations in the LLM; `PYTHON` is a legacy alias only.
23. **Prompt Registry is the only prompt loader** — use `PromptManager`; sampling from prompt metadata.
24. **Supervisor never hardcodes agent names** — capabilities → Registry; register new agents without editing Supervisor.

---

# Current Roadmap

## Completed

- RAG + SQL + business analysis + memory + FastAPI + Streamlit + Docker
- Conversational resolver / state repair / long-term memory
- Attachment upload + preview + ADA + matplotlib charts + CJK fonts
- File-column recognition (incl. Chinese) + row-count aggregation
- Attachment metadata persistence across turns (types / filenames / active sheet)
- Attachment QA skip (no SQL retry overwrite of file answers)
- Hybrid RAG + Adaptive Retrieval Confidence Engine
- Evaluation framework + Prompt Experiment Framework
- Enterprise Prompt Registry
- Multi-agent Supervisor + Agent Registry
- Intent Classifier + Data Source Selection + Clarification + Planner + Orchestrator
- Knowledge Graph + SQL Schema Registry/Retriever
- Specialized Python analytics tools
- Quality Assurance + Reflection with bounded retries
- Documentation refresh for durable Cursor memory

## In Progress

- Broader PlanExecutor coverage for Follow_Up / Upload_File without regressing repair
- Hardening chart defaults for high-cardinality Chinese address columns
- Clear stale attachment/DataFrame caches on re-upload in long-lived Docker sessions

## Planned Features

- Stronger SQL templates / validation for common KPIs
- Expand Clarification Manager patterns (region filters, structured date picker)
- Long-term memory deduplication and retention policies
- Optional auth for deployed API
- Cloud deploy configs
- Snowflake / warehouse executor if requested
- Observability metrics beyond file logs
- Optional Plotly interactive charts only if product asks (keep matplotlib default)
- CI GitHub Action running `pytest` on push

## Known Limitations

- openpyxl `read_only` **must not** be reintroduced as primary Excel path
- Large Excel files load fully into memory (ADA tradeoff)
- Hybrid reranker downloads a local model (first run / image build can be slow/large)
- Chinese chart labels need CJK fonts in the runtime
- Streamlit preview buttons need unique keys per message index
- Parallel orchestrator workers must not mutate `AgentState` directly (main thread commits)
- `.env`, uploads, chroma, and `memory.db` are local runtime artifacts — do not commit secrets or large uploads
- Many `datetime.utcnow()` deprecation warnings remain in memory modules
- Clarification patterns are heuristic; novel ambiguous phrasings may still reach tools
- Leading “no” without a substantive replacement ask still enters repair paths (by design for short corrections)

---

# Instructions for Future Cursor Agents

1. **Read `README.md` first** (this file).
2. **Read the latest session(s) in `DEVELOPMENT_LOG.md`** for recent decisions and pitfalls.
3. **Inspect the repository** before coding (`core/`, `tools/`, `agents/`, `prompts/`, `frontend/`, `api/`, `tests/`).
4. **Continue from the current architecture** — extend modules; do not recreate frameworks.
5. **Never recreate existing functionality** (router, hybrid RAG, retrieval confidence, attachment/ADA pipeline, excel_reader, charts, classifier, data source selector, clarification, planner, orchestrator, QA, reflection, knowledge graph, Prompt Registry, multi-agent Registry/Supervisor, file-column matching).
6. **Extend existing modules** instead of replacing them.
7. **Keep documentation updated**:
   - Update `README.md` whenever architecture or current state changes.
   - **Append** a new dated session to `DEVELOPMENT_LOG.md` (never overwrite history).
8. Preserve modular tool boundaries; keep API/UI thin.
9. Do not add Redis, LangGraph, or new databases unless the user explicitly asks. Multi-agent already exists under `agents/` — extend via Registry, do not invent a second framework.
10. Never log or commit real API keys.
11. Run `pytest tests/ -q` after substantive changes.
12. After `docker-compose.yml` volume changes: `docker compose up -d --force-recreate`. After dependency/Dockerfile changes: rebuild images.
13. If documentation and code disagree, tell the user before making changes.
14. Wait for approval before large redesigns unless the user already requested implementation.
15. When fixing attachment bugs: verify routing (`ATTACHMENT_*`), ADA column match, and that Reflection/QA does not run on file answers.

---

*Last updated: 2026-07-26 — branch `new_feature_1` (attachment metadata persistence across turns).*
