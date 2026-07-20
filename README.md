# GOFO Operations Intelligence Agent

This README is the permanent source of truth for the **current** state of the project.  
Historical engineering sessions live in [`DEVELOPMENT_LOG.md`](./DEVELOPMENT_LOG.md).

**Future Cursor agents: read this file first, then inspect the repository, then continue from the architecture and roadmap below. Do not recreate completed modules.**

---

# Project Overview

## Purpose

`gofo_agent` is an AI **operations intelligence copilot** for GOFO logistics teams. It answers SOP/policy questions, analyzes pickup operations from a SQLite warehouse, investigates root causes, detects anomalies, supports multi-turn operational conversation, and analyzes uploaded files (Excel/CSV/PDF/images) with ChatGPT-style Advanced Data Analysis (ADA) behavior.

## Business Goal

Give operations users one chat surface for:

- SOP / policy questions (RAG over Chroma)
- Live operational metrics (SQL over SQLite)
- Multi-turn follow-ups and corrections (“Actually I mean lowest driver”)
- Uploaded logistics exports (preview + analyze + Python-generated charts)

## Current Development Stage

**Active feature branch, production-style local/Docker service.**

- Core agent, API, Streamlit UI, memory, hybrid RAG, attachments, and matplotlib charts are implemented and tested.
- Latest work on this branch: intent classifier + multi-step planning layer (`core/intent_classifier.py`, `core/planner.py`, `core/plan_executor.py`).
- Test status (2026-07-19): **`280 passed`**.

---

# Branch Information

| Item | Value |
|------|--------|
| **Current branch** | `cursor-memory-version` |
| **Base branch** | `main` |
| **Purpose** | Continue GOFO agent development with durable Cursor memory (README + DEVELOPMENT_LOG) while shipping conversational intelligence, attachments, hybrid RAG, and ADA visualization |

## Major Differences from `main`

Relative to the initial checkpoint on `main` / early history, this branch adds:

1. **Centralized intent routing** (`IntentRouter` + `RouteDispatcher`) — attachments no longer permanently hijack the session
2. **End-to-end multimodal attachments** — upload, validate, process, preview, ADA analysis, matplotlib charts
3. **Hybrid RAG** — Chroma dense + BM25 + RRF + cross-encoder rerank (`BAAI/bge-reranker-base`)
4. **Conversational pipeline** — resolver, semantic orchestration, intent classifier, context builders
5. **Excel robustness** — shared `excel_reader` that avoids openpyxl `read_only` truncation on logistics exports
6. **Docker source mounts + CJK fonts** for live code and Chinese chart labels
7. **Session-aware FastAPI + Streamlit chat composer** with attachment preview

---

# System Architecture

## High-Level Request Flow

```text
User (Streamlit / CLI)
  → FastAPI (session_id → SessionManager → GOFOAgent)
  → optional AttachmentService.process (upload)
  → IntentClassifier.classify(question, ConversationMemory)   # primary intent + tool flags
  → IntentRouter.route(...)   # attachment activate/detach + WAIT_FOR_UPLOAD
  → Planner.plan(question, classification, memory)            # ExecutionPlan (no answers)
  → ToolOrchestrator.run(plan)   # sequential / parallel waves + AgentState
       └─ PlanExecutor adapts to orchestrator (or legacy sequential fallback)
       └─ else RouteDispatcher (single-route legacy path)
  → Generator answer (SQL/RAG/LLM synthesis)
  → ReflectionAgent.critique(answer, evidence)   # never writes the final answer
       └── uses QualityAssurance validators (evidence / reasoning / completeness)
  → Decision: approve | retry retrieval/SQL/python | ask user
  → optional Planner.plan_retry → PlanExecutor → regenerate (max REFLECTION_MAX_RETRIES)
  → ConversationMemory + ConversationState + AttachmentMemory update
  → QueryResponse / AskResponse (answer, sql, data, kpi, charts, sources, plan, quality_report)
  → Streamlit renders text + KPI + chart PNGs
```

Planning and execution are separated: the Planner never answers the user and never runs tools.
QA never generates answers and never executes tools — it only returns a `QualityReport`.
There is **no LangGraph** dependency; stages are modular so a future graph could wrap the same modules.

## Frontend

| Component | Path | Role |
|-----------|------|------|
| Primary UI | `frontend/app.py` | Streamlit **GOFO Operations Intelligence Agent** chat; bottom composer; KPIs; matplotlib chart images |
| Attachment UI | `frontend/attachment_ui.py` | Upload “+” control, history attachment chips, preview buttons (stable keys) |
| Preview | `frontend/attachment_preview.py` | Modal preview for CSV/Excel/PDF/DOCX/text/image using shared Excel reader |
| Legacy UI | `ui/app.py` | Older Streamlit client; not primary |

Frontend talks only to FastAPI (`API_BASE_URL`, default `http://localhost:8000`). It does **not** import SQL/RAG internals.

## Backend / API

| Component | Path | Role |
|-----------|------|------|
| FastAPI app | `api/server.py` | `/health`, `/ask`, `/upload` (multipart); shared `AttachmentService`; session agents |
| Schemas | `api/schemas.py` | `AskRequest`, `AskResponse` (includes `charts`), upload responses |
| Sessions | `core/session.py` | Maps `session_id` → long-lived `GOFOAgent` |

**Important:** Session agents share the process-wide `AttachmentService` so uploads and asks use the same in-memory index (fixes stale attachment state).

## AI Agent

| Component | Path | Role |
|-----------|------|------|
| Agent facade | `core/agent.py` | `GOFOAgent.ask()` — classify → plan → execute/dispatch |
| Intent classifier | `core/intent_classifier.py` | GPT + heuristics → `IntentClassification` (primary intent, confidence, tool flags) |
| Planner | `core/planner.py` | Multi-step `ExecutionPlan` / `ExecutionStep` (never executes) |
| Tool orchestrator | `core/tool_orchestrator.py` | Executes plans with parallel waves, deps, retries, AgentState |
| Plan executor | `core/plan_executor.py` | Adapter to ToolOrchestrator (+ legacy sequential fallback) |
| Reflection (critic) | `core/reflection.py` | Self-critique layer; returns `ReflectionResult` for Planner retries |
| Quality assurance | `core/quality_assurance.py` | Evidence / reasoning / completeness validators + DecisionEngine |
| Intent router | `core/intent_router.py` | Attachment activate/detach + WAIT_FOR_UPLOAD + legacy single-route decisions |
| Dispatcher | `core/route_dispatcher.py` | Executes one route when planner execution is not used |
| Models | `core/models.py` | `QueryResponse` including `charts`, `execution_plan`, `intent_classification` |
| Errors / logging | `core/errors.py`, `error_handler.py`, `logger.py` | Production error types and logging |

Supporting conversational layers:

- `tools/conversation/resolver.py` — corrections / follow-up rewrite
- `tools/orchestration/semantic_analyzer.py` — semantic request analysis
- `tools/planner/intent_classifier.py` — business intent classification
- `tools/context/` — conversation + file context builders
- `tools/router.py` — SQL/RAG/multi-tool planning for analytics paths
- `tools/synthesizer.py` — answer synthesis helpers

## RAG Pipeline

```text
docs/ → ingest.py → Chroma embeddings + bm25_corpus.json
query → tools/rag/retriever.py
         → hybrid.py (dense + BM25 → RRF → optional cross-encoder rerank)
         → generator.py → grounded SOP answer
```

| Module | Role |
|--------|------|
| `tools/rag/retriever.py` | Public `retrieve()` API (hybrid when enabled) |
| `tools/rag/hybrid.py` | Dense + BM25 + RRF + rerank |
| `tools/rag/bm25_index.py` | BM25 index load/query |
| `tools/rag/corpus.py` | Corpus helpers for BM25 |
| `tools/rag/generator.py` | LLM answer from retrieved chunks |
| `tools/rag/rewriter.py` | Query rewrite helpers |
| `ingest.py` | PDF/text ingest into Chroma + BM25 artifact |

Config knobs: `HYBRID_*` in `config.py` / `.env`.

## Vector Database

- **ChromaDB** persisted under `chroma_db/`
- Collection name: `gofo_sop` (env `COLLECTION_NAME`)
- Embeddings: OpenAI `text-embedding-3-small` (default)

## Databases

| DB | Path | Purpose |
|----|------|---------|
| Analytics SQLite | `data/gofo_demo.db` | Pickups, drivers, customers, hubs (SQL analytics) |
| Memory SQLite | `data/memory.db` | Conversations, findings, learned patterns |
| Upload store | `data/uploads/` | Attachment bytes + `attachments_index.json` |

## Attachment / ADA Pipeline

```text
Upload → validator/detector → processors (csv/excel/pdf/docx/text/image)
      → ProcessedFileContext (+ full rows for Excel/CSV)
      → DataFrame store / AttachmentMemory
User question → analysis_intent → data_analysis.analyze_dataframe
      → charts.build_charts (matplotlib PNG base64)
      → AskResponse.charts → Streamlit st.image
```

**Excel rule (do not regress):** never use openpyxl `read_only=True` as the primary reader. Logistics exports often advertise `max_row=2` while thousands of rows exist. Use `tools/files/excel_reader.py` (non-read-only openpyxl + pandas fallback) for both preview and analysis.

## External Integrations

- **OpenAI** — chat (`LLM_MODEL`, default `gpt-4o-mini`) and embeddings
- **Hugging Face / sentence-transformers** — `BAAI/bge-reranker-base` for hybrid rerank (local model download)
- No Slack/Lark/Redis/Snowflake in current scope unless explicitly requested

---

# Repository Structure

```text
gofo_agent/
├── api/                 # FastAPI HTTP layer
├── core/                # Agent, routing, sessions, errors, logging
├── frontend/            # Primary Streamlit UI + attachment preview
├── tools/
│   ├── analysis/        # KPI, root cause, anomaly, recommendations
│   ├── analyzer/        # Business reasoner
│   ├── context/         # Conversation/file context builders
│   ├── conversation/    # Multi-turn resolver / repair
│   ├── files/           # Attachments, ADA, Excel reader, matplotlib charts
│   ├── llm/             # OpenAI client wrappers
│   ├── memory/          # Short-term, state, long-term SQLite memory
│   ├── orchestration/   # Semantic request analysis
│   ├── planner/         # Intent classifier + planner models
│   ├── rag/             # Hybrid retrieval + generation
│   ├── sql/             # Schema-aware planner, executor, summarizer
│   ├── router.py        # Multi-tool analytics router
│   └── synthesizer.py
├── cli/                 # Interactive CLI REPL
├── tests/               # Pytest suite (~250 tests)
├── scripts/             # Demo DB + query helpers
├── data/                # SQLite DBs + uploads (runtime)
├── chroma_db/           # Vector store persistence
├── docs/                # SOP source docs for ingest (if present)
├── config.py            # Central env-backed configuration
├── ingest.py            # RAG + BM25 ingest entrypoint
├── agent.py / query.py  # CLI / legacy entry helpers
├── docker-compose.yml   # gofo-api + gofo-ui with source mounts
├── Dockerfile           # Python 3.11 + matplotlib + WenQuanYi CJK font
├── requirements.txt
├── .env.example
├── README.md            # ← you are here (current state)
└── DEVELOPMENT_LOG.md   # Append-only engineering history
```

### Important files (responsibility)

| File | Responsibility |
|------|----------------|
| `core/agent.py` | Orchestrates ask path; must stay the single agent facade |
| `core/intent_classifier.py` | Primary intent classification (GPT + heuristics) |
| `core/planner.py` | Multi-step ExecutionPlan generation (never executes) |
| `core/tool_orchestrator.py` | ToolRegistry / ToolExecutor / parallel orchestration + AgentState |
| `core/plan_executor.py` | Executes planned tool steps (delegates to ToolOrchestrator) |
| `core/reflection.py` | ReflectionAgent / ReflectionResult self-critique (no answer generation) |
| `core/quality_assurance.py` | Multi-stage QA validators + DecisionEngine (no tool execution) |
| `core/intent_router.py` | Attachment activate/detach + legacy single-route decisions |
| `core/route_dispatcher.py` | Executes one route when planner execution is not used |
| `tools/files/excel_reader.py` | Shared robust Excel IO for preview + analysis |
| `tools/files/charts.py` | Matplotlib PNG generation (`image_base64`) |
| `tools/files/data_analysis.py` | ADA-style per-prompt DataFrame analysis |
| `tools/files/dataframe_store.py` | Parse-once in-memory DataFrames |
| `tools/rag/hybrid.py` | Hybrid retrieval implementation |
| `tools/sql/planner.py` | Schema-aware SQL generation |
| `tools/memory/state.py` | Intent/route/attachment state (corrections) |
| `api/server.py` | Thin HTTP; shared attachment service |
| `frontend/app.py` | Chat + chart image rendering |
| `config.py` | All env defaults; hybrid + upload settings |
| `docker-compose.yml` | Bind-mount source; `MPLCONFIGDIR` |

---

# Features

| Feature | Description | Files | Status |
|---------|-------------|-------|--------|
| SOP RAG Q&A | Hybrid retrieve + generate over SOP corpus | `tools/rag/*`, `ingest.py` | **Complete** |
| SQL analytics | Schema-aware planner + executor over demo SQLite | `tools/sql/*` | **Complete** |
| Business analysis | KPI, root cause, anomaly, recommendations | `tools/analysis/*` | **Complete** |
| Conversation memory | Short-term history + entity resolve | `tools/memory/conversation.py` | **Complete** |
| Conversation state / repair | Corrections, ranking direction, route fields | `tools/memory/state.py`, `tools/conversation/` | **Complete** |
| Long-term memory | SQLite conversations/findings/patterns | `tools/memory/long_memory.py`, `database.py` | **Complete** |
| Centralized intent router | Deterministic multi-route dispatch | `core/intent_router.py`, `route_dispatcher.py` | **Complete** |
| Intent classifier | GPT + heuristics primary intent | `core/intent_classifier.py` | **Complete** |
| Multi-step planner | ExecutionPlan with dependencies | `core/planner.py` | **Complete** |
| Plan executor | Tool registry + step execution | `core/plan_executor.py` | **Complete** |
| Tool orchestrator | Sequential/parallel multi-tool execution | `core/tool_orchestrator.py` | **Complete** |
| Quality assurance | Evidence / reasoning / completeness + retries | `core/quality_assurance.py` | **Complete** |
| Reflection critic | Self-critique + planner feedback (no answer generation) | `core/reflection.py` | **Complete** |
| File upload pipeline | Validate → process → memory | `tools/files/*`, `api/server.py` | **Complete** |
| Attachment preview | Sheet-aware Excel/CSV/PDF/image modal | `frontend/attachment_preview.py` | **Complete** |
| ADA file analysis | Fresh analysis per prompt on stored DataFrame | `data_analysis.py`, `analysis_intent.py` | **Complete** |
| Python chart graphics | Matplotlib PNGs in API response | `tools/files/charts.py`, `frontend/app.py` | **Complete** |
| Excel full-row read | Avoid read_only truncation | `excel_reader.py` | **Complete** |
| Hybrid RAG | Dense + BM25 + RRF + rerank | `tools/rag/hybrid.py` | **Complete** |
| FastAPI + sessions | Persistent agent per `session_id` | `api/`, `core/session.py` | **Complete** |
| Streamlit dashboard | Chat composer, KPIs, charts | `frontend/app.py` | **Complete** |
| CLI | Local debug REPL | `cli/repl.py`, `query.py` | **Complete** |
| Docker Compose | API :8000 + UI :8501 | `Dockerfile`, `docker-compose.yml` | **Complete** |

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
| **Data** | pandas, openpyxl, pypdf, python-docx, Pillow |
| **Charts** | matplotlib (Agg backend → PNG base64) |
| **Analytics DB** | SQLite (`data/gofo_demo.db`) |
| **Memory DB** | SQLite (`data/memory.db`) |
| **Infra** | Docker, Docker Compose; source bind mounts |
| **Fonts (charts)** | `fonts-wqy-zenhei` in image for Chinese labels |
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

Ensure analytics DB exists (demo):

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
| `HYBRID_RERANKER_MODEL` | Default `BAAI/bge-reranker-base` |
| `DEBUG` | Print classifier/planner debug dumps |
| `INTENT_CLASSIFIER_ENABLED` | Run `core.intent_classifier` (default true) |
| `PLANNER_ENABLED` | Execute multi-step plans via `PlanExecutor` (default true; tests disable by default) |
| `QUALITY_ASSURANCE_ENABLED` | Run QA pipeline after generation (default true; tests disable by default) |
| `QA_MAX_RETRIES` | Max planner retries from QA (default `2`) |
| `QA_SCORE_THRESHOLD` | Retry threshold per dimension (default `0.75`) |
| `QA_APPROVE_THRESHOLD` | Immediate-approve threshold (default `0.85`) |
| `REFLECTION_ENABLED` | Run ReflectionAgent after generation (default true; tests disable by default) |
| `REFLECTION_USE_LLM` | Optional LLM critic merge (default false) |
| `REFLECTION_MAX_RETRIES` | Max reflection-driven planner retries (default `2`) |
| `TOOL_ORCHESTRATOR_ENABLED` | Use ToolOrchestrator for plan execution (default true) |
| `TOOL_ORCHESTRATOR_PARALLEL` | Run independent steps in parallel (default true) |
| `TOOL_ORCHESTRATOR_MAX_WORKERS` | Thread pool size for parallel waves (default `4`) |
| `API_BASE_URL` | Streamlit → API (Compose sets `http://gofo-api:8000`) |
| `MPLCONFIGDIR` | Writable matplotlib cache (Compose: `/tmp/matplotlib`) |

**Never commit `.env` or real API keys.**

## Backend startup (local)

```bash
export PYTHONPATH=.
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend startup (local)

```bash
export PYTHONPATH=.
export API_BASE_URL=http://localhost:8000
streamlit run frontend/app.py --server.port 8501
```

## Docker commands

```bash
# Build and start API (:8000) + UI (:8501)
docker compose up --build -d

# Logs
docker compose logs -f gofo-api gofo-ui

# Recreate after docker-compose.yml mount changes
docker compose up -d --force-recreate

# Stop
docker compose down
```

**Compose notes:**

- Source dirs (`api`, `core`, `tools`, `frontend`, …) are bind-mounted so Excel/chart fixes apply without rebuilding for every edit.
- After adding Python deps (e.g. matplotlib), **rebuild the image**: `docker compose build && docker compose up -d`.
- Image includes WenQuanYi Zen Hei for Chinese matplotlib labels.

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

---

# Current Architecture Decisions

These are intentional. Future agents should **not** reverse them without an explicit product request.

1. **Single-agent modular tools, not multi-agent / LangGraph** — keep one `GOFOAgent` and tool modules.
2. **Thin API/UI** — all intelligence lives under `core/` + `tools/`.
3. **Centralized IntentRouter** — attachment mode is session-scoped and can detach on SQL/chat; do not permanently hijack the session after upload.
3b. **Classify → Plan → Execute** — `IntentClassifier` decides *what*; `Planner` decides *how* (steps only); `PlanExecutor` / `RouteDispatcher` run tools. Planner never answers users.
3c. **QA evaluates, never answers** — Evidence / Reasoning / Completeness validators + DecisionEngine; retries go back through Planner (max 2).
3d. **Reflection is the primary critic** — `ReflectionAgent` critiques drafts only; feedback goes to `Planner.plan_retry` (never executes tools).
3e. **ToolOrchestrator executes plans** — Planner decides WHAT; orchestrator decides HOW (deps, parallel waves, retries). No planning inside the orchestrator.
4. **Shared AttachmentService in API process** — session agents must reuse the same upload index.
5. **Parse-once ADA** — store DataFrame in memory; run fresh `analyze_dataframe` per prompt (not canned reuse).
6. **Matplotlib server-side charts** — return `image_base64` PNGs; UI renders with `st.image`. Do not go back to Streamlit-native charts as the primary path.
7. **Excel: no primary `read_only`** — use `tools/files/excel_reader.py` for preview and analysis.
8. **Hybrid RAG behind `retrieve()`** — preserve the public API; dense+BM25+RRF+rerank is the default when enabled.
9. **SQLite only for analytics + memory** — no Redis/Postgres unless requested.
10. **Docker source mounts for iteration** — recreate containers when compose volumes change; rebuild when requirements/Dockerfile change.
11. **Documentation split** — `README.md` = current state; `DEVELOPMENT_LOG.md` = append-only history.

---

# Current Roadmap

## Completed

- RAG + SQL + business analysis + memory + FastAPI + Streamlit + Docker
- Conversational resolver / state repair / long-term memory
- Attachment upload + preview + ADA analysis
- Hybrid RAG (dense + BM25 + RRF + rerank)
- Centralized intent routing
- Intent classifier (`core/intent_classifier.py`) + multi-step planner/executor
- Multi-stage Quality Assurance pipeline (`core/quality_assurance.py`) with bounded retries
- Reflection / self-critique layer (`core/reflection.py`) feeding Planner retries
- Tool Orchestrator (`core/tool_orchestrator.py`) for sequential/parallel multi-tool plans
- Excel full-row reader
- Matplotlib visualization pipeline + CJK fonts in Docker
- Documentation refresh (this README + DEVELOPMENT_LOG)

## In Progress

- Broaden PlanExecutor coverage for Follow_Up / Upload_File without regressing conversation repair
- Hardening chart defaults for Chinese logistics columns (city/status/driver) after multi-row loads
- Clearing stale attachment/DataFrame caches across long-lived Docker sessions when re-uploading the same logical file

## Planned Features

- Stronger SQL templates / validation for common KPIs
- Long-term memory deduplication and retention policies
- Optional auth for deployed API
- Cloud deploy configs
- Snowflake (or warehouse) executor if requested
- Observability metrics beyond file logs

## Known Limitations

- openpyxl `read_only` **must not** be reintroduced as primary Excel path
- Large Excel files load fully into memory (ADA tradeoff)
- Hybrid reranker downloads a local model (first run / image build can be slow/large)
- Chinese chart labels need CJK fonts in the runtime (baked into Dockerfile)
- Streamlit preview buttons need unique keys per message index (already fixed; do not regress)
- `.env`, uploads, chroma, and `memory.db` are local runtime artifacts — do not commit secrets or large uploads

---

# Instructions for Future Cursor Agents

1. **Read `README.md` first** (this file).
2. **Read the latest session in `DEVELOPMENT_LOG.md`** for recent decisions and pitfalls.
3. **Inspect the repository** before coding (`core/`, `tools/`, `frontend/`, `api/`, `tests/`).
4. **Continue from the current architecture** — extend modules; do not recreate frameworks.
5. **Never recreate existing functionality** (router, hybrid RAG, attachment pipeline, excel_reader, charts).
6. **Extend existing modules** instead of replacing them.
7. **Keep documentation updated**:
   - Update `README.md` whenever architecture or current state changes.
   - **Append** a new dated session to `DEVELOPMENT_LOG.md` (never overwrite history).
8. Preserve modular tool boundaries; keep API/UI thin.
9. Do not add Redis, LangGraph, multi-agent systems, or new databases unless the user explicitly asks.
10. Never log or commit real API keys.
11. Run `pytest tests/ -q` after substantive changes.
12. After `docker-compose.yml` volume changes: `docker compose up -d --force-recreate`. After dependency/Dockerfile changes: rebuild images.

---

*Last updated: 2026-07-19 — branch `cursor-memory-version` (classifier + planner + orchestrator + QA + reflection).*
