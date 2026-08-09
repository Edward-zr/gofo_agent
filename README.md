# GOFO Operations Intelligence Agent

This README is the permanent source of truth for the **current** state of this branch.  
Historical engineering sessions live in [`DEVELOPMENT_LOG.md`](./DEVELOPMENT_LOG.md).

**Future Cursor agents: read this file first, then inspect the repository, then continue from the architecture and roadmap below. Do not recreate completed modules.**

---

# Project Overview

## Purpose

`gofo_agent` is an AI **operations intelligence copilot** for GOFO logistics teams. It answers SOP/policy questions, analyzes pickup operations from SQLite, supports multi-turn conversation, and analyzes uploaded files (Excel/CSV/PDF/images) with ChatGPT-style Advanced Data Analysis (ADA)—including **real file column recognition** (English and Chinese headers) and **newest-upload preference** when multiple files exist in a session.

## Business Goal

One chat surface for:

- SOP / policy Q&A (hybrid RAG over Chroma)
- Live operational metrics (SQL over SQLite)
- Multi-turn follow-ups and corrections
- Uploaded logistics / spreadsheet analysis + matplotlib charts
- Multi-step analytical workflows with bounded critique/retries

## Current Development Stage

**Feature branch tip with dual orchestration paths (legacy `GOFOAgent.ask` + optional LangGraph frontier loop).**

| Item | Value |
|------|--------|
| **Branch tip** | `71e2463` — Prefer newest upload for ADA follow-ups |
| **Prior tip** | `960609a` — LangGraph frontier + SOP misroute fixes |
| **Tests** | Attachment newest-upload + SOP misroute + frontier + LangGraph suites green; conftest keeps LangGraph **off** for most tests |
| **Runtime** | FastAPI `:8000` + Streamlit `:8501` via Docker Compose (`--reload` + source mounts) |

**Two ask paths (do not confuse them):**

| Path | When | Behavior |
|------|------|----------|
| **Legacy** | `LANGGRAPH_ENABLED=false` (**code default**) | Full `GOFOAgent.ask` (classifier → router → planner/dispatcher → reflection/QA → memory) |
| **LangGraph** | `LANGGRAPH_ENABLED=true` | Frontier decision tree → TaskSpec → Planner → PlanExecutor → verify (1 replan) → summarize — **no nested `ask`** |

Phase-1 SOP misroute fixes (router, clarification cancel, selector, classifier) apply to **both** paths. Prefer LangGraph for the frontier architecture once soak-tested; keep the flag as a kill switch.

---

# Branch Information

| Item | Value |
|------|--------|
| **Current branch** | `add_langgraph_react_harness` |
| **Tracks** | `origin/add_langgraph_react_harness` |
| **Base / ancestry** | Evolved from `new_feature_1` / earlier feature history; diverged from early `main` |
| **Purpose** | Conversational GOFO intelligence: routing, planner, multi-agent, Prompt Registry, reliable ADA, optional LangGraph frontier orchestration |

## Major Differences from `main`

1. IntentRouter + RouteDispatcher (attachment activate/detach, SOP/SQL/ADA transitions)
2. Intent Classifier + Data Source Selector + Clarification Manager + Planner
3. PlanExecutor / ToolOrchestrator + multi-agent Supervisor + Agent Registry
4. Prompt Registry (`prompts/`)
5. Reflection + QA (skipped for file ADA)
6. Hybrid RAG (Chroma + BM25 + RRF + rerank + confidence)
7. Full attachment/ADA pipeline + Chinese columns + chart types
8. Persisted attachment metadata across turns
9. **LangGraph package** (`graph/`) — frontier → SOP/ADA/SQL tools → summarize
10. **TaskSpec** + **RequestVerifier** (SOP never SQL-retries)
11. **Newest-upload preference** in attachment memory / file reference resolution
12. Docs split: README = current; DEVELOPMENT_LOG = append-only history

---

# System Architecture

## High-Level Request Flow

```text
User (Streamlit / CLI)
  → FastAPI SessionManager → GOFOAgent | LangGraphGOFOAgent
  → optional POST /attachments → AttachmentService.upload + AttachmentMemory.register
  → [LANGGRAPH_ENABLED=true]
       prepare → intent_understanding → frontier
         ├─ not_meaningful → finalize
         └─ task_decomposition → planner → (clarify|tool_execution)
              → result_verification → (replan×1|summarize) → finalize
       Frontier order: SOP searchable? → ADA/analysis? → SQL|Python tools → summarize
  → [LANGGRAPH_ENABLED=false]
       GOFOAgent.ask: IntentClassifier → IntentRouter → Planner|Dispatcher
         → Clarification → PlanExecutor/Supervisor → Reflection/QA → Memory
  → AskResponse (answer, sql, data, kpi, charts, sources, plan, verification, …)
  → Streamlit renders text + KPIs + chart PNGs
```

### Frontier decision order (LangGraph)

```text
1. Understand problem type
2. SOP-related and searchable in SOPs? → RAG (SOP sub-agent)
3. Else ADA/data question?
     No  → not meaningful / out of scope (no SQL date prompt)
     Yes → requires analysis?
           No  → not meaningful
           Yes → choose SQL and/or Python (file ADA) tools → run → summarize
```

### Separation of duties

| Role | Module | Responsibility |
|------|--------|----------------|
| Classify | `IntentClassifier` | What the user wants |
| Route (session) | `IntentRouter` / `RouteDispatcher` | Attachment session + route intent |
| Decompose | `core/task_decomposition.py` | `TaskSpec` constraints |
| Select sources | `DataSourceSelector` | Minimum data sources |
| Plan | `Planner` | Steps/capabilities (never executes) |
| Clarify | `ClarificationManager` | Ask before tools; cancel on new domain ask |
| Frontier | `graph/frontier.py` | Problem-type gate on LangGraph path |
| Supervise | `SupervisorAgent` + Registry | Capability → agent dispatch |
| Execute | `PlanExecutor` / ToolOrchestrator | Run plan steps |
| Verify | `core/request_verifier.py` | TaskSpec checks; **SOP ≠ SQL retry** |
| Critique | Reflection + QA | Legacy ask path (skip file ADA) |
| ADA | `tools/files/*` | Parse-once DataFrame; charts; newest file focus |
| RAG | `tools/rag/*` | Hybrid retrieval + generation |
| SQL | `tools/sql/*` | Schema retrieve → generate → validate → execute |

## Frontend

| Component | Path | Role |
|-----------|------|------|
| Primary UI | `frontend/app.py` | Streamlit chat, upload, KPIs, charts |
| Attachment UI | `frontend/attachment_ui.py` | Pending chips, preview keys |
| Preview | `frontend/attachment_preview.py` | Modal preview for file types |
| Legacy UI | `ui/app.py` | Older client (not primary) |

Frontend talks only to FastAPI (`API_BASE_URL`, default `http://localhost:8000`). Ask body field is **`attachments`** (list of IDs).

## Backend / API

| Component | Path | Role |
|-----------|------|------|
| FastAPI | `api/server.py` | `/health`, `/`, `POST /attachments`, `POST /ask` |
| Schemas | `api/schemas.py` | AskRequest / AskResponse |
| Sessions | `core/session.py` | `session_id` → long-lived agent |

When LangGraph is enabled, `create_session_agent` wraps `GOFOAgent` in `LangGraphGOFOAgent`.

## AI Agent

| Component | Path | Role |
|-----------|------|------|
| Facade | `core/agent.py` | Legacy full pipeline |
| LangGraph runtime | `graph/runtime.py` | Same `ask()` signature |
| Classifier / Router / Planner / Clarifier | `core/*` | Control plane |
| TaskSpec / Verifier | `task_decomposition.py`, `request_verifier.py` | Constraints + SOP-safe verify |
| Multi-agent | `agents/` | RAG, SQL, analytics, reflection, … |

## RAG Pipeline

- Ingest: `ingest.py` → Chroma collection `gofo_sop` (default)
- Retrieve: dense (OpenAI embeddings) + BM25 + RRF + optional cross-encoder rerank
- Confidence engine gates weak retrieval
- Generate grounded SOP answers; sources returned to UI

## Vector Database

- **Chroma** persisted under `chroma_db/` (`CHROMA_PERSIST_DIR`)

## Database

- **SQLite ops warehouse:** `SQLITE_DATABASE` (default `data/gofo_demo.db`)
- **Conversation memory SQLite:** `MEMORY_DATABASE` (default `data/memory.db`)
- Uploads on disk under `data/uploads/`

## External Integrations

- **OpenAI** — chat (`LLM_MODEL`, default `gpt-4o-mini`) + embeddings (`text-embedding-3-small`)
- Optional **LangSmith** when `LANGGRAPH_TRACING` / LangChain tracing env is set
- Local **sentence-transformers** reranker (hybrid RAG)

---

# Repository Structure

```text
gofo_agent/
├── api/                 # FastAPI HTTP layer
├── agents/              # Multi-agent Registry + specialized agents
├── graph/               # LangGraph frontier + planner loop (optional)
├── core/                # Agent facade, router, planner, QA, prompts, TaskSpec, verifier
├── tools/               # RAG, SQL, files/ADA, memory, conversation, KG, charts
├── frontend/            # Primary Streamlit UI
├── prompts/             # Versioned Prompt Registry assets
├── tests/               # Pytest suite
├── evaluation/          # Prompt experiments / datasets
├── docs/                # Source SOP documents for ingest
├── data/                # SQLite DBs + uploads
├── chroma_db/           # Chroma persistence
├── assets/fonts/        # CJK fonts for charts
├── cli/                 # CLI helpers
├── config.py            # All env-backed settings
├── docker-compose.yml   # API + UI with source mounts
├── Dockerfile
├── ingest.py            # SOP ingest into Chroma
├── requirements.txt
├── README.md            # THIS FILE — current state
└── DEVELOPMENT_LOG.md   # Append-only history
```

### Important modules (selected)

| Path | Responsibility |
|------|----------------|
| `graph/frontier.py` | SOP → ADA/analysis → SQL/Python / not_meaningful |
| `graph/nodes.py` / `builder.py` / `runtime.py` | Graph nodes, compile, session wrapper |
| `graph/execution.py` | PlanExecutor + memory/format without nested ask |
| `core/task_decomposition.py` | Heuristic `TaskSpec` |
| `core/request_verifier.py` | VerificationReport; SOP never suggests SQL retry |
| `core/intent_router.py` | Route + detach/chart + hybrid file compares |
| `core/clarification_manager.py` | Clarify; **cancel** on new SOP/domain ask |
| `tools/files/attachment_memory.py` | Active set; **newest-upload focus** |
| `tools/context/file_context_builder.py` | File reference resolution (newest vs compare) |
| `tools/files/charts.py` | Matplotlib charts; honor preferred chart type |
| `tools/rag/*` | Hybrid retrieval + generation |
| `tools/sql/*` | Schema-aware SQL answering |

---

# Features

| Feature | Description | Files | Status |
|---------|-------------|-------|--------|
| SOP hybrid RAG | Dense + BM25 + RRF + rerank + confidence | `tools/rag/*`, `ingest.py` | **Complete** |
| SQL analytics | Schema retriever → SQL → execute | `tools/sql/*` | **Complete** |
| Intent routing | ADA/SOP/SQL/chat; detach; chart type | `core/intent_router.py` | **Complete** |
| Clarification | Ask before tools; cancel on new domain ask | `core/clarification_manager.py` | **Complete** |
| Planner + PlanExecutor | Multi-step plans; Supervisor when multi-agent on | `core/planner.py`, `plan_executor.py`, `agents/` | **Complete** |
| Prompt Registry | Versioned prompts under `prompts/` | `core/prompt_*.py`, `prompts/` | **Complete** |
| Reflection / QA | Bounded retries; skip file ADA | `core/reflection.py`, `quality_assurance.py` | **Complete** |
| File ADA | Upload, preview, column match (EN/ZH), charts | `tools/files/*`, `frontend/*` | **Complete** |
| Newest upload focus | Later file wins for inspect/chart; compare keeps multi | `attachment_memory.py`, `file_context_builder.py` | **Complete** |
| SOP misroute fix | Procedural/CBT follow-ups → RAG not SQL date prompt | router, classifier, selector, clarifier | **Complete** |
| TaskSpec + verifier | Constraints + SOP-safe verification | `task_decomposition.py`, `request_verifier.py` | **Complete** |
| LangGraph frontier loop | Frontier → plan → execute → verify → summarize | `graph/*` | **Complete** (flagged) |
| Docker live reload | Source mounts for api/core/tools/agents/graph/frontend | `docker-compose.yml` | **Complete** |

---

# Tech Stack

| Area | Choice |
|------|--------|
| Languages | Python 3.11+ (dev often 3.13) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Orchestration | LangChain; optional LangGraph |
| AI models | OpenAI chat + embeddings; local BGE reranker |
| Data | pandas, openpyxl, pypdf, python-docx, Pillow |
| Charts | matplotlib (+ CJK fonts under `assets/fonts`) |
| Vector DB | Chroma |
| SQL / memory | SQLite |
| Validation | Pydantic v2 |
| Tests | pytest |
| Infra | Docker + docker-compose |

---

# Installation

## Local setup

```bash
cd gofo_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set OPENAI_API_KEY
# Optional: ingest SOPs
python ingest.py
```

## Environment variables

See [`.env.example`](./.env.example). Critical:

| Variable | Notes |
|----------|--------|
| `OPENAI_API_KEY` | Required |
| `LLM_MODEL` / `EMBEDDING_MODEL` | Defaults `gpt-4o-mini` / `text-embedding-3-small` |
| `SQLITE_DATABASE` / `MEMORY_DATABASE` | Ops + memory DBs |
| `CHROMA_PERSIST_DIR` / `COLLECTION_NAME` | Vector store |
| `LANGGRAPH_ENABLED` | **Code default `false`** in `config.py`; set `true` to use frontier loop |
| `MULTI_AGENT_*`, `PLANNER_*`, `CLARIFICATION_*`, `REFLECTION_*`, `QA_*` | Control plane toggles |

## Backend

```bash
export PYTHONPATH=.
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend

```bash
export API_BASE_URL=http://localhost:8000
streamlit run frontend/app.py --server.port 8501
```

## Docker

```bash
docker compose up --build
# API http://localhost:8000  UI http://localhost:8501
```

Compose mounts `./graph`, `./core`, `./tools`, `./agents`, `./frontend`, etc., for live reload.

---

# Current Architecture Decisions

| Decision | Why | Do not |
|----------|-----|--------|
| Dual path: legacy ask + LangGraph flag | Safe migration; kill switch | Remove legacy until LangGraph soak is done |
| `LANGGRAPH_ENABLED` default **false** in `config.py` | Production safety | Flip default on without UI validation |
| No nested `ask` on LangGraph path | Avoid dual repair/route ownership | Reintroduce `ask(pre_routed=…)` as the tool runner |
| Frontier SOP-first decision order | Fixes CBT/driver SOP misroutes to SQL dates | Ask for “today/this week” on glossary SOP asks |
| Verifier: SOP never SQL-retries | Prevents `SELECT UNKNOWN` overwriting RAG | Re-enable Reflection SQL retry for SOP on graph path |
| Clarification cancels on new domain ask | Stops “tell me more about CBT” resuming a date prompt | Always glue free-text into pending clarification |
| Newest upload is ADA focus | Second file was shadowed by first | Re-activate *all* `processed_contexts` on every turn |
| Hybrid file↔DB/SOP stays on ADA | “Compare it with today’s database” / “follow our SOP” | Detach to pure SQL/SOP mid file session for hybrid compares |
| Prompt Registry only | Versioned prompts | Hardcode prompt strings / temperatures in tools |
| File ADA skips SQL QA | ADA is not SQL | Run SQL Reflection on file charts |
| README vs DEVELOPMENT_LOG | Current vs history | Overwrite DEVELOPMENT_LOG |

---

# Current Roadmap

## Completed

- Control plane: classifier, selector, planner, clarification, orchestrator, multi-agent, prompts
- Hybrid RAG + SQL + ADA (EN/ZH columns, charts, metadata persistence)
- LangGraph frontier loop + TaskSpec + SOP-safe verifier
- SOP misroute / clarification fixes (CBT, driver responsibility, tell me more)
- Newest-upload preference for multi-file sessions

## In Progress / soak

- Validate LangGraph (`LANGGRAPH_ENABLED=true`) in UI against SOP + multi-file ADA scenarios
- Decide when to make LangGraph the day-to-day default (keep flag)

## Planned

- Optional LLM enrichment inside frontier (today: structured heuristics)
- Durable LangGraph checkpointer beyond in-memory MemorySaver
- Stronger column grounding for Chinese headers in charts (e.g. `最新轨迹`)
- UI capability-aware panels (hide SQL panels on pure SOP answers)

## Known Limitations

- `.env.example` may show `LANGGRAPH_ENABLED=true` for local experiments; **code default remains false**
- Frontier v1 is heuristic (not a full LLM “workflow inventor”)
- Multi-file **compare** keeps multiple actives; single-file asks prefer newest
- Legacy Reflection can still SQL-retry SOP on the **legacy** path if QA fires (graph verifier blocks that class of bug)

---

# Instructions for Future Cursor Agents

1. **Read `README.md` first.**
2. **Inspect the repository** before coding (do not rely on chat memory alone).
3. **Continue from the current architecture** — extend `core/`, `tools/`, `graph/`, `agents/`, `frontend/`.
4. **Never recreate** existing router, RAG, ADA, planner, registry, or graph packages from scratch.
5. **Extend modules** instead of replacing them; keep API/UI contracts stable.
6. **Keep documentation updated** — rewrite README for current state; **append** DEVELOPMENT_LOG.
7. **Update README.md whenever architecture or flags change.**
8. When changing LangGraph: keep default-off unless user asks; never expose `reasoning_trace` to end users; SOP verifier must not request SQL retry.
9. When changing attachments: never wipe persisted metadata with empty values; prefer newest upload for inspect/chart; keep hybrid compare paths on ADA.
10. Do not commit secrets, `__pycache__`, `logs/`, or large runtime DBs unless explicitly requested.

---

*Last updated: 2026-08-09 — tip `71e2463` (newest-upload ADA focus + frontier/SOP docs).*
