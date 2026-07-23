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

- SOP / policy questions (hybrid RAG over Chroma)
- Live operational metrics (SQL over SQLite)
- Multi-turn follow-ups and corrections (“Actually I mean lowest driver”)
- Uploaded logistics exports (preview + analyze + Python-generated charts)
- Multi-step analytical workflows with quality critique and bounded retries

## Current Development Stage

**Active feature branch with a production-style local/Docker service and a multi-stage agent pipeline.**

- Core agent, FastAPI, Streamlit UI, memory, hybrid RAG, attachments, matplotlib charts are implemented.
- Newer control plane: Intent Classifier → Planner → Tool Orchestrator → Generator → Reflection/QA (bounded retries).
- Latest branch tip: `4ad63a8` — *Add new feature* (classifier, planner, orchestrator, QA, reflection layers).
- Latest: Multi-Agent Architecture (`agents/` Supervisor + Registry). Test status: **`381 passed`**.

---

# Branch Information

| Item | Value |
|------|--------|
| **Current branch** | `new_feature_1` |
| **Tracks** | `origin/new_feature_1` |
| **Historical base** | Evolved from `cursor-memory-version` / `main` checkpoint history |
| **Purpose** | Ship conversational GOFO intelligence with durable docs plus multi-step planning, orchestration, and self-critique |

## Major Differences from `main`

Relative to the early `main` checkpoint, this line of development adds:

1. **Centralized intent routing** (`IntentRouter` + `RouteDispatcher`) with attachment activate/detach
2. **GPT + heuristic Intent Classifier** (`core/intent_classifier.py`) — richer primary intents than SOP/SQL/General
3. **Data Source Selection** (`core/data_source_selector.py`) — minimum SQL/KG/RAG/Memory/Python before orchestration
4. **Clarification Manager** (`core/clarification_manager.py`) — ask for missing business params before tools; resume plan after answer
5. **Multi-step Planner** (`core/planner.py`) — `ExecutionPlan` / `ExecutionStep` (never answers, never runs tools)
6. **Knowledge Graph** (`tools/knowledge_graph/`) — Driver→Hub→Region, Manager→Hub, SOP ownership
7. **Tool Orchestrator** (`core/tool_orchestrator.py`) — sequential/parallel waves, deps, retries, `AgentState`
8. **Reflection + Quality Assurance** — critique answers; Planner retries (max 2); never generate final text in critics
9. **End-to-end multimodal attachments** — upload, preview, ADA, matplotlib charts
10. **Hybrid RAG** — Chroma dense + BM25 + RRF + cross-encoder rerank
11. **Adaptive Retrieval Confidence Engine** (`tools/rag/confidence.py`) — multi-signal score before generation; Planner-controlled fallbacks
12. **Excel robustness** — shared `excel_reader` avoiding openpyxl `read_only` truncation
13. **Docker source mounts + CJK fonts** for live iteration and Chinese chart labels
14. **Documentation split** — README = current state; DEVELOPMENT_LOG = append-only history
15. **Enterprise Prompt Registry** — versioned prompt assets under `prompts/`; all LLM calls go through PromptManager
16. **Multi-Agent Architecture** — Supervisor + Agent Registry dispatch specialized agents by capability (not agent names)
---

# System Architecture

## High-Level Request Flow

```text
User (Streamlit / CLI)
  → FastAPI (session_id → SessionManager → GOFOAgent)
  → optional AttachmentService.process (upload)
  → IntentClassifier.classify(question, ConversationMemory)
  → IntentRouter.route(...)          # attachment activate/detach + WAIT_FOR_UPLOAD
  → DataSourceSelector.select(...)   # minimum sources (SQL / KG / RAG / Memory / Python)
  → Planner.plan(...)                # ExecutionPlan + required_capabilities
  → ClarificationManager.evaluate(...)  # ask if metric/scope/time missing; else continue
       └─ (user answers) → resume enriched question + original draft plan
  → Supervisor / Agent Registry      # capability → specialized agents (parallel when independent)
       └─ else ToolOrchestrator / PlanExecutor  # legacy when MULTI_AGENT_ENABLED=false
  → Generator / synthesis (SQL/RAG/KG/LLM/ADA answer)
  → ReflectionAgent.critique(...)    # primary critic (uses QA validators)
       └── QualityAssurancePipeline  # evidence / reasoning / completeness
  → Decision: approve | plan_retry | ask_user  (max REFLECTION_MAX_RETRIES = 2)
  → ConversationMemory + ConversationState + AttachmentMemory update
  → AskResponse (answer, sql, data, kpi, charts, sources, plan, reflection, agent_state)
  → Streamlit renders text + KPI + chart PNGs
```

**Separation of duties (do not collapse these):**

| Role | Module | Responsibility |
|------|--------|----------------|
| Classify | `IntentClassifier` | *What* the user wants |
| Select sources | `DataSourceSelector` | *Which* data sources are necessary (min set) |
| Plan | `Planner` | *What steps* to run (never executes); emits capabilities |
| Clarify | `ClarificationManager` | *Ask* for missing business params before tools |
| Supervise | `SupervisorAgent` + `AgentRegistry` | *Which agents* by capability; dispatch/merge (no business logic) |
| Orchestrate | `ToolOrchestrator` | Legacy *how* when `MULTI_AGENT_ENABLED=false` |
| Critique | `ReflectionAgent` + QA validators | Evaluate draft answers (never write final answers) |
| Route (legacy/session) | `IntentRouter` / `RouteDispatcher` | Attachment session semantics + single-route fallback |

There is **no LangGraph** dependency. Stages are modular so a future graph could wrap the same modules.

## Clarification Manager

Sits **after Planner, before Tool Orchestrator**. Prevents hallucinations by refusing to guess missing business parameters.

| Ambiguous question | Asks for | Example options |
|--------------------|----------|-----------------|
| "Show the best driver." | Ranking metric | Completed pickups / Pickup rate / On-time rate / Customer rating |
| "Show performance." | Entity scope | By hub / By driver / By region / Overall network |
| "Compare pickup rates." | Time periods | Today vs Yesterday / This week vs Last week / … |
| "Show top customers." | Ranking metric | Volume / Revenue / Pickup count |

**Rules**
- Never guess missing metrics, scopes, or time ranges.
- Ask only when required; skip when conversation memory already has the slot or confidence + question are already rich.
- Multiple-choice when possible; user may reply with option number or label.
- Pending clarification is stored on the session `GOFOAgent` (`pending_clarification`) and mirrored into `AgentState` / response (`original_question`, `missing_fields`, `pending_question`, `user_response`).
- After the user answers, the original draft plan resumes with an enriched `resolved_question` — conversation does not restart.

**Debug:** set `DEBUG=true` or `CLARIFICATION_DEBUG=true` to print missing parameters, reason, and resumed execution plan.

## Data Source Selection Rules
| Source | Use when |
|--------|----------|
| **SQL** | Metrics, counts, KPIs, aggregations, rankings, historical ops data |
| **Knowledge Graph** | Relationships: Driver → Hub → Region, Manager → Hub, SOP ownership |
| **RAG** | Unstructured SOP / policy / document explanations |
| **Memory** | Follow-ups and conversation context |
| **Python** | Calculations, statistics, forecasting, or chart data prep only |
| **Visualization** | Explicit chart / plot / dashboard requests |

Sources are combined **only when necessary** (e.g. SQL+RAG for “compare rate and explain reasons”, SQL+KG for “who manages the lowest-rate hub”, SQL+Python for trend charts).

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
| FastAPI app | `api/server.py` | `/health`, `/ask`, `/upload`; shared `AttachmentService`; session agents |
| Schemas | `api/schemas.py` | `AskRequest`, `AskResponse` (includes `charts`) |
| Sessions | `core/session.py` | `session_id` → long-lived `GOFOAgent` |

**Important:** Session agents share the process-wide `AttachmentService` so uploads and asks use the same in-memory index.

## AI Agent

| Component | Path | Role |
|-----------|------|------|
| Agent facade | `core/agent.py` | `GOFOAgent.ask()` — full pipeline orchestration |
| Intent classifier | `core/intent_classifier.py` | Primary intent + tool flags + confidence |
| Data source selector | `core/data_source_selector.py` | Minimum SQL/KG/RAG/Memory/Python sources |
| Planner | `core/planner.py` | Multi-step `ExecutionPlan` / `plan_retry` |
| Plan executor | `core/plan_executor.py` | Adapter to ToolOrchestrator (+ legacy sequential fallback) |
| Tool orchestrator | `core/tool_orchestrator.py` | Parallel waves, deps, AgentState, tool registry |
| Quality assurance | `core/quality_assurance.py` | Evidence / reasoning / completeness validators |
| Reflection | `core/reflection.py` | Self-critique API for Planner retries |
| Intent router | `core/intent_router.py` | Attachment activate/detach + WAIT_FOR_UPLOAD |
| Dispatcher | `core/route_dispatcher.py` | Single-route execution fallback |
| Models | `core/models.py` | `QueryResponse` (+ plan, QA, reflection, agent_state) |

Supporting layers under `tools/`: conversation resolver, semantic orchestration, business analysis, SQL (schema registry + retriever), RAG, knowledge graph, files/ADA, memory.

## Prompt Registry

Every LLM interaction loads prompts through the **Prompt Registry** (single source of truth). Prompts are versioned software assets with YAML frontmatter — not ad-hoc markdown strings in Python.

```text
Component
  → PromptManager.get / render / invoke
  → PromptRegistry (discover, version, cache, active selection)
  → PromptRenderer ({{variables}} + required-variable validation)
  → LLM client (temperature / max_tokens / model from prompt metadata)
```

| Module | Path | Role |
|--------|------|------|
| Registry | `core/prompt_registry.py` | Discover/load/cache; active / candidate / experimental; hot reload; overrides |
| Manager | `core/prompt_manager.py` | Only API components should use; `invoke` applies metadata to LLM |
| Renderer | `core/prompt_renderer.py` | Template vars; missing-required errors |
| Validator | `core/prompt_validator.py` | Frontmatter parse + metadata schema |
| Experiments | `core/prompt_experiments.py` | Multi-version eval + recommend best by priorities |

**Asset layout** (`prompts/`):

```text
prompts/
├── router/ | planner/ | sql/ | rag/ | python/
├── recommendation/ | reflection/ | clarification/
├── shared/          # system_rules, formatting_rules, business_rules
└── experiments.yaml
```

Each family has `*_vN.md` + `registry.yaml` (`active`, optional `candidate` / `experimental`). Switch without code changes:

```bash
# Session override
export PROMPT_EXPERIMENT=planner.planner_prompt:v2

# Or persist into registry.yaml
python -m evaluation.prompt_experiments --activate planner.planner_prompt:v2 --persist
python -m evaluation.prompt_experiments --list
python -m evaluation.prompt_experiments --mode mock
```

**Debug:** `DEBUG=true` or `PROMPT_DEBUG=true` prints name, version, metadata, variables, missing vars, size/tokens, experiment.

**Rules for contributors:** never `open()` prompt files from tools; never hardcode temperature/max_tokens at call sites — use PromptManager / metadata.

## Multi-Agent Architecture

Planner emits **capabilities**, not agent names. The **Supervisor** asks the **Agent Registry** which agents can run each task, dispatches (optionally in parallel), merges standardized responses, and returns. The Supervisor contains **no business logic**.

```text
Planner (required_capabilities)
  → Supervisor Agent
  → Agent Registry (discover / health / select)
  → RAG | SQL | Analytics | General | Memory | Reflection | Recommendation
  → Supervisor merge → QueryResponse
```

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

Each agent implements `initialize` / `can_handle` / `execute` / `validate` / `summarize` / `health_check` and returns:

```json
{"agent":"SQL","status":"SUCCESS","confidence":0.94,"result":{...}}
```

Add a new agent by implementing `BaseAgent`, calling `registry.register(agent)` — **do not edit the Supervisor**.

**Flags:** `MULTI_AGENT_ENABLED` (default true), `MULTI_AGENT_PARALLEL`, `MULTI_AGENT_DEBUG`, `MULTI_AGENT_REFLECTION`. When multi-agent is off, PlanExecutor falls back to ToolOrchestrator.

## SQL Analytics Pipeline

```text
Question
  → Business Understanding (intent + date rewrite)
  → Schema Retriever (relevant tables/columns/joins only)
  → SQL Generator (LLM uses retrieved schema ONLY)
  → Validator (reject unknown tables/columns)
  → Executor (SQLite)
  → Summarizer
```

| Component | Path | Role |
|-----------|------|------|
| Schema Registry | `tools/sql/schema_registry.py` | Live tables/columns/PKs/FKs + descriptions; auto-refresh on DB change |
| Schema Retriever | `tools/sql/schema_retriever.py` | Minimum relevant schema via intent + keywords (+ optional embeddings) |
| SQL Planner/Generator | `tools/sql/planner.py` | LLM SQL from retrieved schema only |
| Validator | `tools/sql/validator.py` | Allowlist gate → `SELECT 'UNKNOWN'` on invent |
| Service | `tools/sql/service.py` | End-to-end SQL capability |

Refresh after migrations: `from tools.sql import refresh_schema_registry; refresh_schema_registry()`  
Debug: set `DEBUG=true` or `SCHEMA_RETRIEVER_DEBUG=true` to print intent, selected tables/columns, and joins.

## RAG Pipeline

```text
docs/ → ingest.py → Chroma embeddings + bm25_corpus.json
query → Retriever (hybrid dense + BM25 → RRF → rerank)
      → Retrieval Confidence Engine (multi-signal adaptive score)
      → Decision Engine (Planner retrieval_policy)
      → Generator (HIGH confident / MEDIUM cautious / LOW fallback)
```

| Component | Path | Role |
|-----------|------|------|
| Retriever | `tools/rag/retriever.py` / `hybrid.py` | Dense + BM25 + RRF + optional rerank |
| Confidence Engine | `tools/rag/confidence.py` | Weighted score from similarity, chunk count, diversity, metadata |
| Service | `tools/rag/service.py` | retrieve → confidence → decide → generate |
| Generator | `tools/rag/generator.py` | Grounded SOP answer; adapts prompts to confidence level |

**Confidence levels:** HIGH (≥0.80) confident generate · MEDIUM (0.60–0.79) cautious generate · LOW (&lt;0.60) never confident — clarify / request docs / general knowledge only if Planner allows.

**Planner policy** on `ExecutionPlan.retrieval_policy`:

```json
{
  "allow_general_knowledge": false,
  "minimum_confidence": "MEDIUM",
  "allow_clarification": true,
  "allow_document_request": true
}
```

Debug: `DEBUG=true` or `RETRIEVAL_CONFIDENCE_DEBUG=true` prints scores, level, strategy, and policy. Tune weights via `RETRIEVAL_CONF_W_*` env vars (normalized at runtime).

## Vector Database

- **ChromaDB** under `chroma_db/`
- Collection: `gofo_sop` (`COLLECTION_NAME`)
- Embeddings: OpenAI `text-embedding-3-small` (default)
- Reranker: `BAAI/bge-reranker-base` via `sentence-transformers`

## Databases

| DB | Path | Purpose |
|----|------|---------|
| Analytics SQLite | `data/gofo_demo.db` | Pickups, drivers, customers, hubs |
| Memory SQLite | `data/memory.db` | Conversations, findings, patterns |
| Knowledge Graph | `tools/knowledge_graph/` | Driver→Hub→Region, Manager→Hub, SOP ownership (SQLite drivers + org maps) |
| Upload store | `data/uploads/` | Attachment bytes + `attachments_index.json` |

## Attachment / ADA Pipeline

```text
upload → AttachmentService
       → detector/validator/processor
       → dataframe_store (parse once)
       → analyzer / data_analysis (ADA)
       → charts.py (matplotlib PNG base64)
```

## Specialized Python Analytics Tools

```text
SQL rows
  → TransformationTool   (filter/sort/groupby/pivot/clean)
  → StatisticsTool       (mean/median/ratios/growth/ranking/correlation)
  → VisualizationTool    (line/bar/pie/scatter/histogram)
  → RecommendationTool   (business insights from computed results)
  → Generator (LLM explains outputs; never calculates)
```

| Tool | Path | Responsibility |
|------|------|----------------|
| Transformation | `tools/python/transformation_tool.py` | Prepare DataFrames |
| Statistics | `tools/python/statistics_tool.py` | Compute metrics |
| Visualization | `tools/python/visualization_tool.py` | Chart metadata + PNG images |
| Recommendation | `tools/python/recommendation_tool.py` | Ops recommendations (not raw SQL) |

`ToolName.PYTHON` remains a backward-compatible alias to StatisticsTool. AgentState stores `dataframe`, `transformed_dataframe`, `statistics`, `chart_metadata`, `recommendations`.

Attachment ADA still uses `tools/files/charts.py` directly:

```text
Upload → validate/detect → processors → ProcessedFileContext
      → DataFrame store / AttachmentMemory
User question → analysis_intent → analyze_dataframe
      → charts.build_charts (matplotlib PNG base64)
      → AskResponse.charts → Streamlit st.image
```

**Excel rule (do not regress):** never use openpyxl `read_only=True` as the primary reader. Use `tools/files/excel_reader.py`.

## External Integrations

- **OpenAI** — chat (`LLM_MODEL`, default `gpt-4o-mini`) and embeddings
- **Hugging Face / sentence-transformers** — local reranker model
- No Slack/Lark/Redis/Snowflake/LangGraph in current scope unless explicitly requested

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
│   ├── conversation/    # Multi-turn resolver / repair
│   ├── files/           # Attachments, ADA, Excel reader, matplotlib charts
│   ├── llm/             # OpenAI client wrappers
│   ├── memory/          # Short-term, state, long-term SQLite memory
│   ├── orchestration/   # Semantic request analysis
│   ├── planner/         # Legacy capability planner + business Intent enum
│   ├── python/          # Specialized analytics: transform/stats/viz/recommend
│   ├── knowledge_graph/ # Driver→Hub→Region, Manager→Hub, SOP ownership
│   ├── rag/             # Hybrid retrieval + generation
│   ├── sql/             # Schema registry/retriever, planner, executor, validator
│   ├── router.py        # Multi-tool analytics router
│   └── synthesizer.py
├── cli/                 # Interactive CLI REPL
├── evaluation/          # Benchmark + scenario evaluation framework
├── tests/               # Pytest suite
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
| `core/agent.py` | Single agent facade; wires full pipeline |
| `core/intent_classifier.py` | Primary intent classification (GPT + heuristics) |
| `core/data_source_selector.py` | Minimum necessary data sources before planning |
| `core/planner.py` | Multi-step ExecutionPlan + `plan_retry` (never executes) |
| `core/tool_orchestrator.py` | HOW to execute plans (parallel/deps/retries/AgentState) |
| `core/plan_executor.py` | Delegates to orchestrator; keeps mock-friendly sequential path |
| `core/quality_assurance.py` | Evidence / reasoning / completeness validators |
| `core/reflection.py` | Product-facing critic API |
| `core/prompt_registry.py` | Prompt discovery, versions, active selection |
| `core/prompt_manager.py` | Component API for render/invoke |
| `core/intent_router.py` | Attachment activate/detach; WAIT_FOR_UPLOAD |
| `core/route_dispatcher.py` | Single-route fallback execution |
| `tools/knowledge_graph/service.py` | Relationship lookups (hub/region/manager/SOP) |
| `tools/sql/schema_registry.py` | Live DB metadata registry + refresh |
| `tools/sql/schema_retriever.py` | Relevant schema subset for SQL LLM prompts |
| `tools/sql/validator.py` | Reject invented tables/columns |
| `tools/files/excel_reader.py` | Shared robust Excel IO |
| `tools/files/charts.py` | Matplotlib PNG generation |
| `tools/rag/hybrid.py` | Hybrid retrieval |
| `api/server.py` | Thin HTTP; shared attachment service |
| `frontend/app.py` | Chat + chart rendering |
| `config.py` | All env defaults (classifier/planner/QA/reflection/orchestrator) |

---

# Features

| Feature | Description | Files | Status |
|---------|-------------|-------|--------|
| SOP hybrid RAG | Dense + BM25 + RRF + rerank | `tools/rag/*`, `ingest.py` | **Complete** |
| SQL analytics | Schema retriever → generate → validate → execute | `tools/sql/*` | **Complete** |
| Schema registry/retriever | Only relevant tables/columns to SQL LLM | `tools/sql/schema_registry.py`, `schema_retriever.py` | **Complete** |
| Specialized Python tools | Transform / Statistics / Visualization / Recommendation | `tools/python/*` | **Complete** |
| Business analysis | KPI, root cause, anomaly, recommendations | `tools/analysis/*` | **Complete** |
| Conversation memory | Short-term history + entity resolve | `tools/memory/*` | **Complete** |
| Conversation repair | Corrections, ranking direction | `tools/conversation/`, `tools/memory/state.py` | **Complete** |
| Long-term memory | SQLite conversations/findings/patterns | `tools/memory/long_memory.py` | **Complete** |
| Intent router | Attachment session routing | `core/intent_router.py` | **Complete** |
| Intent classifier | Primary intents + confidence/tool flags | `core/intent_classifier.py` | **Complete** |
| Data source selection | Minimum necessary SQL/KG/RAG/Memory/Python | `core/data_source_selector.py` | **Complete** |
| Multi-step planner | ExecutionPlan with selected sources + deps | `core/planner.py` | **Complete** |
| Knowledge graph | Driver/hub/region/manager/SOP ownership | `tools/knowledge_graph/` | **Complete** |
| Tool orchestrator | Parallel/sequential multi-tool execution | `core/tool_orchestrator.py` | **Complete** |
| Quality assurance | Evidence/reasoning/completeness | `core/quality_assurance.py` | **Complete** |
| Reflection critic | Self-critique + bounded Planner retries | `core/reflection.py` | **Complete** |
| Prompt Registry | Versioned prompts, metadata, render, experiments | `core/prompt_*.py`, `prompts/` | **Complete** |
| Multi-agent Supervisor | Capability routing via Agent Registry | `agents/` | **Complete** |
| File upload + ADA | Validate → process → analyze → charts | `tools/files/*` | **Complete** |
| Attachment preview | Sheet-aware modal preview | `frontend/attachment_preview.py` | **Complete** |
| Excel full-row read | Anti-truncation reader | `tools/files/excel_reader.py` | **Complete** |
| FastAPI + sessions | Persistent agent per `session_id` | `api/`, `core/session.py` | **Complete** |
| Streamlit UI | Chat composer, KPIs, charts | `frontend/app.py` | **Complete** |
| CLI | Local debug REPL | `cli/`, `query.py` | **Complete** |
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
| **Orchestration libs** | LangChain OpenAI client wrappers (not LangGraph) |
| **Data** | pandas, openpyxl, pypdf, python-docx, Pillow |
| **Charts** | matplotlib (Agg → PNG base64) |
| **Analytics DB** | SQLite (`data/gofo_demo.db`) |
| **Memory DB** | SQLite (`data/memory.db`) |
| **Infra** | Docker, Docker Compose; source bind mounts |
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
| `DEBUG` | Classifier/planner/QA/orchestrator debug dumps |
| `INTENT_CLASSIFIER_ENABLED` | Run primary IntentClassifier |
| `PLANNER_ENABLED` | Multi-step planner execution path |
| `CLARIFICATION_MANAGER_ENABLED` | Ask for missing business params before tools (default true) |
| `CLARIFICATION_DEBUG` | Print missing params / reason / resumed plan |
| `RETRIEVAL_CONFIDENCE_ENABLED` | Adaptive multi-signal retrieval confidence (default true) |
| `RETRIEVAL_CONFIDENCE_DEBUG` | Print retrieval confidence breakdown + fallback strategy |
| `RETRIEVAL_CONFIDENCE_HIGH` / `RETRIEVAL_CONFIDENCE_MEDIUM` | Level cutoffs (default 0.80 / 0.60) |
| `DATA_SOURCE_SELECTION_ENABLED` | Minimum data-source selection before planning (default true) |
| `SCHEMA_RETRIEVER_ENABLED` | Retrieve relevant schema before SQL generation (default true) |
| `SCHEMA_RETRIEVER_DEBUG` | Print retrieved tables/columns/joins |
| `SCHEMA_EMBEDDING_RETRIEVAL_ENABLED` | Optional vector search over schema metadata (default false) |
| `PYTHON_TOOLS_DEBUG` | Debug specialized Python analytics tools |
| `QUALITY_ASSURANCE_ENABLED` | QA validators (also used by Reflection) |
| `QA_MAX_RETRIES` / `QA_SCORE_THRESHOLD` / `QA_APPROVE_THRESHOLD` | QA thresholds |
| `REFLECTION_ENABLED` / `REFLECTION_USE_LLM` / `REFLECTION_MAX_RETRIES` | Self-critique loop |
| `TOOL_ORCHESTRATOR_ENABLED` / `TOOL_ORCHESTRATOR_PARALLEL` / `TOOL_ORCHESTRATOR_MAX_WORKERS` | Plan execution |
| `PROMPT_REGISTRY_DIR` | Prompt asset root (default `prompts`) |
| `PROMPT_HOT_RELOAD` | Reload registry when prompt files change |
| `PROMPT_DEBUG` | Print prompt selection / render debug |
| `PROMPT_EXPERIMENT` | Force version `key:version` (e.g. `planner.planner_prompt:v2`) |
| `MULTI_AGENT_ENABLED` | Supervisor + Agent Registry path (default true) |
| `MULTI_AGENT_PARALLEL` | Concurrent independent agent tasks |
| `MULTI_AGENT_MAX_WORKERS` | Thread pool size for parallel agents |
| `MULTI_AGENT_DEBUG` | Print capability selection / dispatch / merge |
| `MULTI_AGENT_REFLECTION` | Optional Reflection agent pass after tools |
| `API_BASE_URL` | Streamlit → API |
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
docker compose up --build -d
docker compose logs -f gofo-api gofo-ui
docker compose up -d --force-recreate   # after compose volume changes
docker compose down
```

**Compose notes:** source dirs are bind-mounted; rebuild after `requirements.txt` / Dockerfile changes.

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

Integration-heavy agent paths disable planner/QA/reflection/clarification execution by default in `tests/conftest.py`. Unit tests cover those modules directly.

## Evaluation Framework

Comprehensive regression suite under `evaluation/`:

| Path | Role |
|------|------|
| `evaluation/datasets/*.json` | ~200 SOP, 100 SQL, 50 general, 50 follow-up, 30 memory, 20 upload, 40 scenarios |
| `evaluation/runner.py` | Independent benchmark questions |
| `evaluation/scenario_runner.py` | Multi-turn workflows |
| `evaluation/evaluator.py` / `metrics.py` | Functional, performance, cost, reliability scoring |
| `evaluation/report.py` | `evaluation/results/latest_report.md` |
| `evaluation/compare.py` | Regression vs previous combined run |

**One command:**

```bash
export PYTHONPATH=.
python -m evaluation                  # offline mock smoke (default)
python -m evaluation --mode live      # real GOFOAgent (needs keys/DB)
python -m evaluation --limit 20       # sample
python -m evaluation --generate-datasets
```

Scores semantic correctness, tool usage, planner/orchestration, latency, tokens/cost — not exact answer match. Results JSON + markdown report land in `evaluation/results/`.

### Prompt experiments

Benchmark each prompt version via the evaluation framework; compare accuracy, SQL success, scenario success, latency, tokens, and estimated cost; recommend the best version by configurable priorities.

```bash
export PYTHONPATH=.
python -m evaluation.prompt_experiments --mode mock
python -m evaluation.prompt_experiments --experiment exp_planner_prompt
```

Reports are written under `evaluation/results/` (see `core/prompt_experiments.py`).

---

# Current Architecture Decisions

These are intentional. Future agents should **not** reverse them without an explicit product request.

1. **Single-agent modular tools, not multi-agent / LangGraph** — one `GOFOAgent`; stages are node-ready but no LangGraph dependency.
2. **Thin API/UI** — intelligence lives under `core/` + `tools/`.
3. **Classify → Select sources → Plan → Orchestrate → Critique** — do not collapse Planner into Executor or let critics write final answers.
4. **Planner never executes tools; Orchestrator never plans; DataSourceSelector never executes.**
5. **Minimum necessary sources** — do not call every tool; combine SQL/KG/RAG/Python only when the question requires it.
6. **Reflection is the primary critic**; QA validators are the multi-dimension scoring engine underneath.
7. **Bounded retries (default 2)** — never infinite critique loops.
8. **Centralized IntentRouter still owns attachment activate/detach** — uploads must not permanently hijack the session.
9. **Shared AttachmentService in API process** — session agents reuse the same upload index.
10. **Parse-once ADA** — store DataFrame; fresh `analyze_dataframe` per prompt.
11. **Matplotlib server-side charts** — `image_base64` PNGs; UI uses `st.image`.
12. **Excel: no primary `read_only`** — use `tools/files/excel_reader.py`.
13. **Hybrid RAG behind `retrieve()`** — preserve public API.
14. **SQLite only for analytics + memory** — no Redis/Postgres unless requested.
15. **Docker source mounts for iteration** — recreate on volume changes; rebuild on dependency changes.
16. **Documentation split** — README = current; DEVELOPMENT_LOG = append-only history.
18. **SQL Schema Retriever** — never dump the full schema into the LLM prompt unless explicitly requested; validate against the registry allowlist.
20. **Specialized Python tools** — never put calculations in the LLM; use TRANSFORM → STATISTICS → VISUALIZATION → RECOMMENDATION. `PYTHON` is a legacy alias only.
21. **Business `classifier_intent` stays on the legacy `tools.planner.intent_classifier.Intent` values** for API compatibility; new taxonomy lives in `intent_classification` / `primary_intent`.
22. **Prompt Registry is the only prompt loader** — components use `PromptManager`; LLM sampling comes from prompt metadata; active versions switch via `registry.yaml` / `PROMPT_EXPERIMENT`, not code edits.
23. **Supervisor never hardcodes agent names** — Planner emits `required_capabilities`; Agent Registry selects implementations; new agents register without Supervisor changes.

---

# Current Roadmap

## Completed

- RAG + SQL + business analysis + memory + FastAPI + Streamlit + Docker
- Conversational resolver / state repair / long-term memory
- Attachment upload + preview + ADA + matplotlib charts + CJK fonts
- Hybrid RAG (dense + BM25 + RRF + rerank)
- Adaptive Retrieval Confidence Engine (multi-signal score + Planner fallbacks)
- Evaluation framework (benchmark + multi-turn scenarios, report + regression compare)
- Enterprise Prompt Registry + Prompt Experiment Framework
- Multi-agent Supervisor + Agent Registry (capability routing)
- Centralized intent routing + shared AttachmentService
- Intent Classifier + Data Source Selection + Clarification Manager + multi-step Planner + Tool Orchestrator
- Knowledge Graph relationships (Driver→Hub→Region, Manager→Hub, SOP ownership)
- SQL Schema Registry + Retriever (relevant schema only before generation)
- Specialized Python analytics tools (transform/stats/viz/recommend)
- Quality Assurance validators + Reflection critic with bounded retries
- Documentation refresh for durable Cursor memory

## In Progress

- Broaden PlanExecutor/Orchestrator coverage for Follow_Up / Upload_File without regressing conversation repair
- Hardening chart defaults for Chinese logistics columns after multi-row loads
- Clearing stale attachment/DataFrame caches across long-lived Docker sessions on re-upload

## Planned Features

- Stronger SQL templates / validation for common KPIs
- Expand Clarification Manager patterns (region filters, custom date ranges via structured picker)
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

---

# Instructions for Future Cursor Agents

1. **Read `README.md` first** (this file).
2. **Read the latest session(s) in `DEVELOPMENT_LOG.md`** for recent decisions and pitfalls.
3. **Inspect the repository** before coding (`core/`, `tools/`, `frontend/`, `api/`, `tests/`).
4. **Continue from the current architecture** — extend modules; do not recreate frameworks.
5. **Never recreate existing functionality** (router, hybrid RAG, retrieval confidence, attachment pipeline, excel_reader, charts, classifier, data source selector, clarification manager, planner, orchestrator, QA, reflection, knowledge graph, Prompt Registry, multi-agent Registry/Supervisor).
6. **Extend existing modules** instead of replacing them.
7. **Keep documentation updated**:
   - Update `README.md` whenever architecture or current state changes.
   - **Append** a new dated session to `DEVELOPMENT_LOG.md` (never overwrite history).
8. Preserve modular tool boundaries; keep API/UI thin.
9. Do not add Redis, LangGraph, multi-agent systems, or new databases unless the user explicitly asks.
10. Never log or commit real API keys.
11. Run `pytest tests/ -q` after substantive changes.
12. After `docker-compose.yml` volume changes: `docker compose up -d --force-recreate`. After dependency/Dockerfile changes: rebuild images.
13. If documentation and code disagree, tell the user before making changes.
14. Wait for approval before large redesigns unless the user already requested implementation.

---

*Last updated: 2026-07-23 — branch `new_feature_1` (Multi-Agent Architecture).*
