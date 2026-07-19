# GOFO Operations Intelligence Agent

This README is the permanent project memory for `gofo_agent`. A future Cursor Agent with no chat history must read this file first, then inspect the repository, and continue from the current state. This document records the project purpose, architecture, completed work, important decisions, bug history, current status, and next development tasks.

This README replaces previous Cursor conversation memory.

## PROJECT OVERVIEW

### Project Name

`GOFO Operations Intelligence Agent`

Repository folder:

```text
gofo_agent/
```

### Purpose

Build an AI operations intelligence assistant for GOFO logistics teams. The assistant answers SOP questions, analyzes pickup operations, explains operational performance, investigates root causes, detects anomalies, gives recommendations, and supports multi-turn operational conversations through CLI, FastAPI, and Streamlit.

### Business Problem

GOFO operations users need fast answers to questions such as:

- What is CBT?
- What is the pickup SOP?
- How many pickups happened today?
- Which hub performed worst?
- Why did Chicago fail again?
- Which hub does Drew Nguyen belong to?
- Rank all hubs by performance.
- Actually I mean lowest driver.
- Anything abnormal today?

These questions require multiple knowledge sources:

- SOP and policy PDFs, best handled by RAG over ChromaDB.
- Operational metrics, best handled by SQL over SQLite analytics data.
- Conversation context, best handled by short-term session memory and structured conversation state.
- Past issues and recurring patterns, best handled by persistent SQLite long-term memory.

### Final Goal

The system should become a production-style operations intelligence copilot:

- RAG for SOP and policy knowledge.
- SQL analytics for operational data.
- FastAPI backend.
- Streamlit operations dashboard.
- CLI for local debugging.
- Short-term memory for multi-turn conversation.
- Structured state memory for corrections and ranking-direction repair.
- SQLite long-term memory for conversations, findings, and learned patterns.
- Root-cause analysis, anomaly detection, KPI knowledge, and operational recommendations.
- Docker and Docker Compose support.

### Current Progress

The project is functional and has:

- RAG ingestion/retrieval/generation.
- SQLite operational analytics.
- SQL/RAG/multi-tool router.
- KPI overview, root-cause drilldown, anomaly detection, and recommendations.
- Session-aware FastAPI backend.
- Streamlit dashboard.
- CLI with debug mode.
- SQLite long-term memory.
- Structured conversation state repair.
- Production configuration, logging, error handling, Docker, and Compose.
- Current test status from the latest run: `179 passed`.

## SYSTEM ARCHITECTURE

### Current High-Level Data Flow

```text
User
  -> Streamlit frontend OR CLI
  -> FastAPI API layer
  -> SessionManager
  -> GOFOAgent
  -> Attachment upload/process (optional)
  -> Conversation Resolver
  -> Intent Classifier
  -> Context Builder + File Context Builder
  -> Data Source Router (SQLITE / RAG / ATTACHMENT / combined)
  -> Planner / Router OR Attachment Analyzer
  -> SQL Executor and/or SOP RAG and/or File Processors
  -> Business Analyzer
  -> Conversation Memory + Attachment Memory Update
  -> Long-Term Memory retrieval/storage
  -> Structured response
  -> Frontend/API/CLI output
```

### Frontend

Primary frontend:

- `frontend/app.py`
- Streamlit dashboard named **GOFO Operations Intelligence Center**.
- Uses `st.session_state.messages` for UI chat history.
- Generates `st.session_state.session_id` and sends it to FastAPI on every request.
- Calls FastAPI through `API_BASE_URL`, defaulting to `http://localhost:8000`.
- Displays chat answers, KPI cards, root-cause analysis, recommendations, generated SQL, SQL rows, historical memory context, and SOP sources.
- Does not import SQL/RAG/router internals.

Legacy/simple frontend:

- `ui/app.py`
- Earlier Streamlit UI kept for compatibility.
- Calls FastAPI but is not the main dashboard.

### Backend/API Layer

Active backend:

- `api/server.py`
- FastAPI app exposing:
  - `GET /health`
  - `GET /`
  - `POST /attachments` (multipart file upload)
  - `POST /ask` (supports optional `attachments: [attachment_id, ...]`)
- Uses `api/schemas.py` for public API models.
- Uses `core.session.SessionManager` to preserve one `GOFOAgent` per `session_id`.
- Has request logging middleware.
- Has centralized exception handlers for `AgentError`, `sqlite3.Error`, and generic errors.
- Returns friendly JSON errors:

```json
{
  "success": false,
  "error": "friendly message"
}
```

Deprecated backend:

- `app.py`
- Legacy Flask retrieval-only API from early development.
- Do not use for new backend work unless explicitly asked.

### AI Agent Layer

There are two agent entrypoints:

- `agent.py`
  - Low-level public application API.
  - `ask(question, result_context=None) -> QueryResponse`.
  - Builds `QueryRequest` and calls `tools.router.route()`.

- `core/agent.py`
  - Production-style wrapper class `GOFOAgent`.
  - Owns session-scoped `ConversationResolver`, `ConversationMemory`, `ConversationState`, `AttachmentMemory`, and `AttachmentService`.
  - Performs:
    - attachment upload processing and file context building,
    - semantic conversation resolution,
    - intent classification,
    - context building,
    - data source routing (SQLITE, RAG, ATTACHMENT, combined),
    - attachment-aware analysis (CSV/Excel/PDF/DOCX/text/image),
    - result-context routing,
    - `agent.ask()` call,
    - business reasoning,
    - semantic and compatibility memory updates,
    - dashboard-friendly response formatting.

Use `core.agent.GOFOAgent` for API/dashboard/production session use. Keep `agent.py` as the low-level core API.

### Session Architecture

FastAPI must not create a new agent per request.

Current correct flow:

```text
Application startup
  -> SessionManager()
  -> POST /ask receives session_id
  -> session_manager.get_session(session_id)
  -> existing GOFOAgent
  -> existing ConversationMemory + ConversationState
```

Files:

- `core/session.py`
- `api/server.py`
- `frontend/app.py`

`SessionManager` stores:

```text
session_id -> GOFOAgent
```

This preserves:

- `ConversationMemory`
- result memory
- entity memory
- repair memory
- structured `ConversationState`

### Conversational Reasoning Flow

Current agent flow:

```text
Question
  -> tools.conversation.ConversationResolver
  -> tools.planner.intent_classifier.classify_intent()
  -> tools.context.build_context()
  -> agent.ask() / tools.router.route()
  -> tools.analyzer.business_reasoner
  -> semantic memory update
  -> compatibility memory update
```

The current production pipeline is:

```text
Question
  -> Conversation Resolver
  -> Intent Classifier
  -> Context Builder
  -> Planner
  -> SQL Executor
  -> Business Analyzer
  -> Conversation Memory Update
  -> Response Formatter
```

New conversational modules:

- `tools/conversation/resolver.py`
  - Session-level semantic state owner.
  - Keeps the last 10 structured turns.
  - Stores original question, resolved question, intent, metric, dimension, entities, filters, SQL, SQL rows, business findings, KPI summary, ranking, best entity, worst entity, root cause, recommendations, summary, and date range.
  - Resolves follow-ups such as `why`, `worst hub`, `best driver`, `compare yesterday`, `show details`, and pronouns.
  - Avoids inheriting stale filters for fresh global ranking questions.

- `tools/planner/intent_classifier.py`
  - Classifies conversation-aware intents:
    - `SQL_QUERY`
    - `SUMMARY`
    - `ROOT_CAUSE`
    - `RECOMMENDATION`
    - `COMPARISON`
    - `TREND`
    - `EXPLANATION`
    - `DRILLDOWN`
    - `FOLLOWUP`
    - `CORRECTION`
    - `CLARIFICATION`
    - `GREETING`
    - `UNKNOWN`

- `tools/context/context_builder.py`
  - Rebuilds missing metric, dimension, date, and target context from semantic memory.
  - Decides when a response can be answered from previous SQL-backed analysis.
  - Allows contextual root-cause/recommendation turns to inherit target filters but keeps fresh rankings global unless the user explicitly specifies a filter.

- `tools/analyzer/business_reasoner.py`
  - Adds evidence-backed executive summaries, operational KPIs, business findings, root-cause metadata, recommendations, and suggested next investigations.
  - Can answer recommendation/explanation follow-ups from previous SQL evidence without generating new SQL.

Compatibility memory modules still exist:

- `tools/memory/conversation.py`
  - Short-term memory, recent turns, active entities, previous SQL/result context.

- `tools/memory/state.py`
  - Structured state memory retained for compatibility with existing tests and debug output.

- `tools/memory/repair.py`
  - Legacy deterministic repair detector retained for CLI/debug compatibility.

- `tools/memory/resolver.py`
  - Legacy LLM-based resolver retained as fallback for existing entity-resolution behavior.

Important behavior:

- `worst hub` resolves from semantic ranking memory, not raw text.
- `What should operations do?` can answer from the previous SQL-backed analysis without new SQL.
- `Compare yesterday` inherits the previous metric/topic and changes only the period.
- `Show details` routes through previous result context.
- Fresh questions like `Which driver has the highest performance?` do not inherit a previous hub filter unless the user explicitly asks for that hub.

Legacy memory flow preserved for compatibility:

```text
Question
  -> Semantic Conversation Resolver
  -> optional ConversationMemory fallback
  -> Router
```

Memory components:

- `tools/memory/conversation.py`
  - Short-term memory, recent turns, active entities, previous SQL/result context.

- `tools/memory/state.py`
  - Structured state memory for intent-level repair.
  - Tracks:
    - `last_question`
    - `last_resolved_question`
    - `last_intent`
    - `last_metric`
    - `last_dimension`
    - `last_sort_direction`
    - `last_filters`
    - `last_entities`
    - `last_sql`
    - `last_result_summary`

- `tools/memory/repair.py`
  - Detects correction phrases such as `actually`, `I mean`, `no`, `wrong`, `instead`.

- `tools/memory/resolver.py`
  - LLM-based follow-up resolver for contextual questions and pronouns.

- `tools/memory/result_analyzer.py`
  - Answers follow-ups using prior SQL result rows only.

Important rule:

- Entity references such as `he`, `she`, `that driver` use entity memory.
- Corrections such as `actually`, `instead`, `I mean`, `wrong` use structured state memory.
- Corrections must not answer from old entity memory.

Examples now supported:

```text
Highest performing driver
-> Drew Nguyen

Which hub does he belong to?
-> Uses entity memory for Drew Nguyen

Actually I mean lowest driver
-> Uses ConversationState
-> Resolves to: Show lowest performing driver
```

```text
Rank all hubs by performance
-> Chicago Hub worst

Why is the worst hub performing badly?
-> Resolves to: Why is Chicago Hub performing badly?
```

### Router and Tool Execution

File:

- `tools/router.py`

Responsibilities:

- Accepts `QueryRequest`.
- If `result_context` exists, routes to memory analysis.
- Infers business context.
- Retrieves long-term memory before analysis.
- Routes special business questions:
  - operations overview -> KPI analysis,
  - abnormal/anomaly questions -> anomaly analysis,
  - why/root-cause questions -> drilldown analysis.
- Otherwise calls natural-language planner.
- Executes SQL and/or RAG.
- Synthesizes multi-tool responses.
- Attaches root cause, anomaly, recommendation, and long-memory metadata.
- Saves important conversations/findings to long-term memory.

Important:

- SQL execution and RAG remain in their own tool modules.
- Router orchestrates; it should not become SQL executor or RAG retriever.

### RAG Pipeline

Files:

- `ingest.py`
- `tools/rag/retriever.py`
- `tools/rag/rewriter.py`
- `tools/rag/generator.py`
- `tools/rag/service.py`

Flow:

```text
PDFs in docs/
  -> ingest.py
  -> text chunks
  -> OpenAI embeddings
  -> ChromaDB collection gofo_sop

Question
  -> rewrite for retrieval
  -> retrieve chunks
  -> confidence filter
  -> grounded generation
  -> QueryResponse(capability="rag")
```

Design decisions:

- Retriever owns all ChromaDB access.
- Generator never retrieves.
- Generator must answer only from SOP context.
- Low-confidence retrieval skips GPT generation.

### Vector Database

ChromaDB is used for SOP vector retrieval.

Current paths:

- `chroma_db/`
- `chroma_db/chroma.sqlite3`

Collection:

- `gofo_sop`

Source docs:

- `docs/CBT 商家大会.pdf`
- `docs/GL揽收分工SOP.pdf`

### Traditional Databases

Operational analytics database:

- `data/gofo_demo.db`
- SQLite.
- Config path: `config.SQLITE_DATABASE`.

Persistent memory database:

- `data/memory.db`
- SQLite.
- Config path: `config.MEMORY_DATABASE`.

Memory tables:

- `conversation_history`
- `operational_findings`
- `learned_patterns`

No Redis, PostgreSQL, Snowflake, or external database is currently used.

### SQL Analytics Pipeline

Files:

- `tools/sql/schema.py`
- `tools/sql/schema_loader.py`
- `tools/sql/planner.py`
- `tools/sql/business_date.py`
- `tools/sql/executor.py`
- `tools/sql/summarizer.py`
- `tools/sql/service.py`
- `tools/sql/formatter.py`

Flow:

```text
Question
  -> latest business date from SQLite
  -> relative date rewrite
  -> SQL planner
  -> SQLite executor
  -> summarizer
  -> QueryResponse(capability="sql")
```

Important schema decisions:

- Operational date is always `pickups.pickup_date`.
- Hub means operational warehouse/station and always maps to `drivers.hub`.
- Hub must never be represented as `customers.customer_name`.
- Driver questions join `drivers`.
- Customer questions join `customers`.
- City/location questions join `addresses`.
- Failure reason questions join `exceptions`.
- Details questions return actual rows, not only counts.

### Business Analysis Layer

Folder:

- `tools/analysis/`

Modules:

- `context.py`
  - Infers metric, dimension, date range, filters, and detects broad operations, why, and anomaly questions.

- `kpi.py`
  - Loads `config/kpi_dictionary.yaml`.
  - Answers broad questions like “How are operations?”.
  - Checks pickup volume, completion rate, failed pickups, delayed pickups, hub ranking, and driver performance.

- `drilldown.py`
  - Runs deterministic multi-level SQL drilldowns for why/root-cause questions.

- `root_cause.py`
  - Produces root-cause metadata from SQL rows.

- `anomaly.py`
  - Uses statistics only: moving average, percentage change, z-score.
  - No ML model.

- `recommender.py`
  - Generates operational recommendations only.
  - Does not execute actions.

### Long-Term Memory and Learning

Files:

- `tools/memory/database.py`
- `tools/memory/long_memory.py`
- `tools/memory/learning.py`
- `tools/memory/findings.py`

Long-term memory uses SQLite only.

Behavior:

- Saves important conversations.
- Saves operational findings.
- Saves repeated learned patterns.
- Searches memory using keyword matching, entity matching, metric matching, and dimension matching.
- Historical memory is supporting evidence only.
- Current database results remain highest priority.

Do not replace current SQL results with old memory.

Correct answer pattern:

```text
Current Analysis:
Current database result...

Historical Context:
Similar issue detected previously...

Comparison:
Current database results remain the source of truth.
```

### Production Infrastructure

Files:

- `.env.example`
- `config.py`
- `core/logger.py`
- `core/errors.py`
- `core/error_handler.py`
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`

Features:

- Environment variable loading through `python-dotenv`.
- Structured logging to console and file.
- Log file path from `LOG_FILE`.
- Logs directory auto-created.
- Centralized friendly error handling.
- FastAPI middleware request logging.
- Docker support for API and UI.
- Docker Compose orchestration.

### External Integrations

Current integrations:

- OpenAI via `langchain-openai`.
- ChromaDB local persistent vector store.
- SQLite local databases.

Not currently implemented:

- LangGraph.
- MCP tools.
- Snowflake.
- Slack/Lark/Feishu/WeChat.
- External auth.
- External hosted databases.
- Multi-agent architecture.
- Forecasting models.
- Action execution system.

## CURRENT PROJECT STRUCTURE

Important actual repository structure:

```text
gofo_agent/
├── README.md
├── README_RUN.md
├── .env
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.py
├── agent.py
├── planner.py
├── query.py
├── ingest.py
├── app.py
├── api/
│   ├── __init__.py
│   ├── schemas.py
│   └── server.py
├── cli/
│   ├── __init__.py
│   ├── colors.py
│   ├── display.py
│   └── repl.py
├── config/
│   └── kpi_dictionary.yaml
├── core/
│   ├── __init__.py
│   ├── agent.py
│   ├── error_handler.py
│   ├── errors.py
│   ├── logger.py
│   ├── models.py
│   └── session.py
├── data/
│   ├── gofo_demo.db
│   └── memory.db
├── docs/
│   ├── CBT 商家大会.pdf
│   └── GL揽收分工SOP.pdf
├── frontend/
│   └── app.py
├── logs/
│   └── agent.log
├── scripts/
│   ├── create_demo_db.py
│   └── test_queries.py
├── tools/
│   ├── analyzer/
│   ├── analysis/
│   ├── context/
│   ├── conversation/
│   ├── llm/
│   ├── memory/
│   ├── planner/
│   ├── rag/
│   ├── sql/
│   ├── router.py
│   └── synthesizer.py
├── ui/
│   └── app.py
└── tests/
```

### Root Files

`README.md`

Purpose:

- Permanent project memory and source of truth.

`README_RUN.md`

Purpose:

- Quick start commands for API, UI, and CLI.

`.env`

Purpose:

- Local environment values.
- Must not be committed or copied into documentation.
- Treat any exposed key as compromised and rotate it.

`.env.example`

Purpose:

- Example environment configuration.
- Should contain placeholders only. Verify it does not contain a real API key before sharing or committing.

`requirements.txt`

Purpose:

- Python dependencies.

`config.py`

Purpose:

- Central environment-backed configuration.
- Loads `.env`.
- Exposes paths and settings:
  - `OPENAI_API_KEY`
  - `LLM_MODEL`
  - `EMBEDDING_MODEL`
  - `SQLITE_DATABASE`
  - `MEMORY_DATABASE`
  - `CHROMA_PERSIST_DIR`
  - `CHROMA_PATH`
  - `COLLECTION_NAME`
  - `API_HOST`
  - `API_PORT`
  - `LOG_LEVEL`
  - `LOG_FILE`
- Raises `ConfigurationError` for missing OpenAI key via `require_openai_api_key()`.

`agent.py`

Purpose:

- Low-level public application API.
- Provides `ask(question, result_context=None) -> QueryResponse`.
- Used by `core.agent.GOFOAgent`.

`planner.py`

Purpose:

- Backward-compatible shim exporting `tools.planner.plan`.

`query.py`

Purpose:

- CLI entry point.
- Uses `core.agent.GOFOAgent` for normal CLI.
- Delegates `--debug` to `cli.repl.run_repl(debug=True)`.

`ingest.py`

Purpose:

- Ingest SOP PDFs into ChromaDB.

`app.py`

Purpose:

- Legacy Flask retrieval-only API.
- Deprecated for new work.

`Dockerfile`

Purpose:

- Builds Python 3.11 slim image.
- Installs requirements.
- Runs FastAPI with Uvicorn.

`docker-compose.yml`

Purpose:

- Orchestrates `gofo-api` and `gofo-ui`.
- Mounts `data/`, `logs/`, and `chroma_db/`.
- Exposes `8000` and `8501`.

### API Folder

`api/server.py`

Purpose:

- FastAPI backend.
- Creates `SessionManager`.
- Routes `POST /ask` by `session_id`.
- Provides `GET /health`.
- Logs requests and session memory debug details.
- Handles errors centrally.

`api/schemas.py`

Purpose:

- Public API request/response models.
- `AskRequest` includes:
  - `question`
  - `session_id`, default `"default"`
- `AskResponse` includes:
  - `answer`
  - `sources`
  - `sql`
  - `data`
  - `analysis`
  - `recommendations`
  - `kpi`
  - `raw`

### CLI Folder

`cli/repl.py`

Purpose:

- Interactive debug REPL.
- Uses `ConversationMemory`, repair detector, and memory resolver.
- Debug mode prints memory/planner/SQL/RAG details.

`cli/display.py`

Purpose:

- Formats CLI output and debug panels.
- Prints memory state, planner state, SQL, rows, root cause, anomaly, recommendations, long memory matches, and learned patterns.

`cli/colors.py`

Purpose:

- Terminal color helpers.

### Core Folder

`core/models.py`

Purpose:

- Shared Pydantic models:
  - `QueryRequest`
  - `SourceChunk`
  - `QueryResponse`
- `QueryResponse` now contains optional debug/analysis/memory fields including classifier intent, inherited context, business findings, semantic memory status, and SQL cache hit status.

`core/agent.py`

Purpose:

- Production wrapper `GOFOAgent`.
- Owns per-session:
  - `ConversationResolver`
  - `ConversationMemory`
  - `ConversationState`
- Runs the production conversational pipeline and converts `QueryResponse` into dashboard/API dictionary response.

`core/session.py`

Purpose:

- `SessionManager`.
- Stores `session_id -> GOFOAgent`.
- Ensures FastAPI sessions preserve memory across requests.

`core/logger.py`

Purpose:

- Structured logger with console and file handlers.
- Creates logs directory automatically.

`core/errors.py`

Purpose:

- Custom exceptions:
  - `AgentError`
  - `DatabaseError`
  - `RetrievalError`
  - `PlannerError`
  - `MemoryError`
  - `ConfigurationError`

`core/error_handler.py`

Purpose:

- Converts exceptions into friendly user/API messages.

### Frontend Folders

`frontend/app.py`

Purpose:

- Main Streamlit dashboard.
- Maintains UI chat history.
- Sends `session_id` to FastAPI.
- Displays structured response sections.

`ui/app.py`

Purpose:

- Older/simple Streamlit UI.
- Kept for compatibility.

### Config Folder

`config/kpi_dictionary.yaml`

Purpose:

- KPI definitions for:
  - `pickup_completion_rate`
  - `delay_rate`
  - `failure_rate`
  - `package_volume`

### Data Folder

`data/gofo_demo.db`

Purpose:

- SQLite demo operational analytics DB.

`data/memory.db`

Purpose:

- SQLite persistent long-term memory DB.

### Docs Folder

`docs/`

Purpose:

- SOP PDFs used for RAG ingestion.

### Tools Folder

`tools/router.py`

Purpose:

- Main orchestration between planner, SQL, RAG, business analysis, memory analysis, and long-term memory.

`tools/synthesizer.py`

Purpose:

- LLM synthesis for hybrid SQL + RAG answers.

`tools/conversation/resolver.py`

Purpose:

- Semantic conversation resolver and memory store.
- Maintains the last 10 structured turns.
- Resolves follow-up references, corrections, ranking references, pronouns, drilldowns, and comparisons.
- Extracts rankings, best entity, worst entity, business findings, recommendations, and date/metric/dimension context from responses.

`tools/context/context_builder.py`

Purpose:

- Builds execution-ready context from the semantic resolver.
- Reconstructs missing metric, dimension, date range, filters, and inherited context.
- Prevents stale filters from leaking into fresh global rankings.

`tools/analyzer/business_reasoner.py`

Purpose:

- Evidence-based business reasoning layer.
- Adds executive summary, operational KPI text, business findings, root cause, recommendations, and suggested next investigation.
- Answers recommendation/explanation follow-ups from prior SQL evidence when no new SQL is needed.

`tools/planner/intent_classifier.py`

Purpose:

- Context-aware intent classifier for conversational operations questions.
- Supports `SQL_QUERY`, `SUMMARY`, `ROOT_CAUSE`, `RECOMMENDATION`, `COMPARISON`, `TREND`, `EXPLANATION`, `DRILLDOWN`, `FOLLOWUP`, `CORRECTION`, `CLARIFICATION`, `GREETING`, and `UNKNOWN`.

`tools/llm/client.py`

Purpose:

- Shared cached `ChatOpenAI`.
- All GPT calls should use this client.

`tools/planner/`

Purpose:

- Natural-language capability planner.

`tools/rag/`

Purpose:

- RAG retrieval, rewriting, generation, and service orchestration.

`tools/sql/`

Purpose:

- SQL schema loading, planning, execution, date resolution, summarization, formatting, and service orchestration.

`tools/memory/`

Purpose:

- Short-term memory, structured conversation state, repair detection, long-term memory, learning, result analysis, and compatibility shims.

`tools/analysis/`

Purpose:

- Business context, KPI analysis, drilldown, root cause, anomaly detection, and recommendations.

### Tests Folder

`tests/`

Purpose:

- Pytest suite covering RAG, SQL, planner, router, synthesizer, CLI/memory behavior, FastAPI, long-term memory, state repair, logging, config, and health API.

Recent result:

```text
128 passed
```

## COMPLETED FEATURES

### SOP PDF Ingestion

Status:

- Done.

Files:

- `ingest.py`
- `docs/`
- `chroma_db/`
- `config.py`

Description:

- Loads SOP PDFs, splits text, embeds chunks, stores vectors in ChromaDB.

How it works:

- Uses LangChain PDF loader and text splitter.
- Uses OpenAI embeddings.
- Stores chunks in collection `gofo_sop`.

### RAG Retrieval

Status:

- Done.

Files:

- `tools/rag/retriever.py`
- `core/models.py`

Description:

- Retrieves SOP chunks from ChromaDB.

How it works:

- Embeds query.
- Queries ChromaDB.
- Converts distance to higher-is-better score.
- Returns `SourceChunk` objects.

### RAG Query Rewriting

Status:

- Done.

Files:

- `tools/rag/rewriter.py`
- `tools/rag/service.py`

Description:

- Rewrites vague SOP questions before retrieval.

How it works:

- Example: `CBT` becomes a fuller CBT question for retrieval.
- Generator still receives the original user question.

### RAG Generation

Status:

- Done.

Files:

- `tools/rag/generator.py`
- `tools/llm/client.py`

Description:

- Produces grounded SOP answers from retrieved chunks.

How it works:

- Uses only SOP context.
- Returns fallback when answer is not in available SOP documents.

### RAG Service

Status:

- Done.

Files:

- `tools/rag/service.py`

Description:

- Orchestrates rewrite, retrieve, confidence filtering, generation.

### Natural-Language Planner

Status:

- Done.

Files:

- `tools/planner/planner.py`
- `tools/planner/models.py`
- `planner.py`

Description:

- Decides `sql`, `rag`, `multi`, or `unknown`.

How it works:

- LLM returns JSON `PlanningDecision`.
- Planner does not answer questions and does not execute tools.

### SQL Analytics Planner

Status:

- Done and upgraded.

Files:

- `tools/sql/planner.py`
- `tools/sql/schema.py`
- `tools/sql/schema_loader.py`

Description:

- Generates SQLite SQL from operational analytics questions.

How it works:

- Injects documented schema and live SQLite metadata.
- Enforces known table/column references.
- Includes business vocabulary and examples.
- Reinforces hub = `drivers.hub`.

### SQL Executor

Status:

- Done.

Files:

- `tools/sql/executor.py`

Description:

- Executes SQLite queries against `data/gofo_demo.db`.

How it works:

- Uses `sqlite3.Row`.
- Returns `list[dict]`.
- Raises clear error if DB is missing.

### SQL Business Date Resolver

Status:

- Done.

Files:

- `tools/sql/business_date.py`

Description:

- Treats latest pickup date in DB as operational today.

How it works:

- Reads `MAX(pickup_date)` from `pickups`.
- Rewrites relative dates before SQL planning.

### SQL Summarizer

Status:

- Done.

Files:

- `tools/sql/summarizer.py`

Description:

- Summarizes SQL rows into concise business answers.

### SQL Service

Status:

- Done.

Files:

- `tools/sql/service.py`

Description:

- Orchestrates business date resolution, SQL planning, execution, and summarization.

### Multi-Tool Routing and Synthesis

Status:

- Done.

Files:

- `tools/router.py`
- `tools/synthesizer.py`

Description:

- Supports SQL-only, RAG-only, multi-tool SQL+RAG, memory analysis, operations overview, anomaly, and why/root-cause paths.

### Demo SQLite Database

Status:

- Done.

Files:

- `scripts/create_demo_db.py`
- `scripts/test_queries.py`
- `data/gofo_demo.db`

Description:

- Demo operational database for local development and tests.

Schema:

- `drivers`
- `customers`
- `addresses`
- `pickups`
- `exceptions`

### Short-Term Conversation Memory

Status:

- Done.

Files:

- `tools/memory/conversation.py`
- `tools/memory/resolver.py`
- `tools/memory/summarizer.py`
- `tools/memory/result_analyzer.py`

Description:

- Tracks recent turns, entities, metrics, filters, SQL context, and result context.

### Structured Conversation State Repair

Status:

- Done.

Files:

- `tools/memory/state.py`
- `core/agent.py`
- `tests/test_state_memory.py`

Description:

- Fixes correction regressions after FastAPI Docker migration.

How it works:

- `ConversationState` tracks intent, metric, dimension, sort direction, filters, SQL, and result summary.
- Corrections like `actually lowest driver` are resolved from state before entity memory can reuse the old driver.

### Production Conversational Reasoning Pipeline

Status:

- Done.

Files:

- `tools/conversation/resolver.py`
- `tools/planner/intent_classifier.py`
- `tools/context/context_builder.py`
- `tools/analyzer/business_reasoner.py`
- `core/agent.py`
- `tests/test_conversational_pipeline.py`

Description:

- Refactors `GOFOAgent` into a staged conversational operations analyst pipeline.
- Semantic memory now stores meaning, including rankings, best/worst entities, filters, SQL rows, findings, recommendations, and summaries.
- Follow-up questions such as `Why?`, `What should operations do?`, `Compare yesterday`, and `Show details` are resolved from structured state.
- Recommendation and explanation follow-ups can be answered from previous SQL evidence without generating new SQL.
- Fresh global ranking questions do not inherit stale entity filters unless explicitly specified.

How it works:

- `ConversationResolver` resolves conversational meaning and records semantic state.
- `IntentClassifier` classifies the user turn with context.
- `ContextBuilder` reconstructs missing metric/dimension/date/filter information.
- Existing `agent.ask()` and `tools.router.route()` still execute SQL/RAG.
- `BusinessReasoner` adds executive summaries, business findings, recommendations, and next-investigation guidance.
- Semantic memory and legacy memory are both updated for backward compatibility.

### Conversation Repair Detector

Status:

- Done.

Files:

- `tools/memory/repair.py`
- `tests/test_conversation_repair.py`

Description:

- Detects correction phrases and simple dimension/ranking/date replacements.

### Result Context Memory

Status:

- Done.

Files:

- `tools/memory/result_analyzer.py`
- `tools/router.py`
- `tools/memory/conversation.py`

Description:

- Lets follow-ups like `rank them`, `sort them`, `lowest one`, `top 5` analyze previous SQL rows without a new DB query.

### Persistent Long-Term Memory

Status:

- Done.

Files:

- `tools/memory/database.py`
- `tools/memory/long_memory.py`
- `tools/memory/learning.py`
- `tools/memory/findings.py`
- `data/memory.db`

Description:

- Stores important conversations, findings, and repeated patterns in SQLite.

How it works:

- Initializes tables automatically.
- Searches by keyword/entity/metric/dimension matching.
- Saves patterns when issues recur.

### KPI Knowledge Layer

Status:

- Done.

Files:

- `config/kpi_dictionary.yaml`
- `tools/analysis/kpi.py`
- `tests/test_kpi.py`

Description:

- Defines pickup completion rate, delay rate, failure rate, and package volume.
- Broad questions like “How are operations?” trigger automatic KPI overview.

### Root Cause Analyzer

Status:

- Done.

Files:

- `tools/analysis/root_cause.py`
- `tools/analysis/drilldown.py`
- `tests/test_business_reasoning.py`

Description:

- Why/root-cause questions trigger deterministic drilldown SQL and root-cause summary.

### Anomaly Detector

Status:

- Done.

Files:

- `tools/analysis/anomaly.py`

Description:

- Detects volume spikes, delay spikes, failure spikes, and completion drops using statistics only.

### Recommendation Engine

Status:

- Done.

Files:

- `tools/analysis/recommender.py`

Description:

- Generates operational recommendations.
- Does not execute actions.

### FastAPI Backend

Status:

- Done and productionized.

Files:

- `api/server.py`
- `api/schemas.py`
- `core/session.py`

Description:

- Provides session-aware `/ask` and `/health`.
- Logs requests and memory debug.
- Uses centralized errors.

### Streamlit Dashboard

Status:

- Done.

Files:

- `frontend/app.py`

Description:

- Main operations intelligence dashboard.

How it works:

- Maintains UI chat history.
- Sends `session_id` with requests.
- Displays KPI, SQL, root cause, recommendations, historical context, and SOP sources.

### CLI

Status:

- Done.

Files:

- `query.py`
- `cli/repl.py`
- `cli/display.py`
- `cli/colors.py`

Description:

- Local CLI and debug mode.

### Production Infrastructure

Status:

- Done.

Files:

- `.env.example`
- `core/logger.py`
- `core/errors.py`
- `core/error_handler.py`
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `tests/test_config.py`
- `tests/test_logging.py`
- `tests/test_health_api.py`

Description:

- Environment config, structured logging, centralized errors, API middleware logging, Docker, Docker Compose.

### Test Suite

Status:

- Done and broad.

Latest known result:

```text
128 passed
```

Run:

```bash
.venv/bin/pytest tests/ -q
python -m pytest tests/ -q
```

## CURRENT TECH STACK

Languages:

- Python
- YAML
- Dockerfile syntax

Frameworks:

- FastAPI
- Streamlit
- Flask legacy only
- Pytest

AI tools:

- OpenAI chat models through `langchain-openai`
- OpenAI embeddings
- LangChain document/text utilities

Databases:

- SQLite for operational analytics
- SQLite for persistent long-term memory
- ChromaDB for vector search

Infrastructure:

- Docker
- Docker Compose
- Uvicorn
- Local filesystem volumes for `data/`, `logs/`, and `chroma_db/`

Python packages:

- `chromadb`
- `openai`
- `python-dotenv`
- `flask`
- `fastapi`
- `uvicorn`
- `streamlit`
- `requests`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-core`
- `langchain-text-splitters`
- `pypdf`
- `pytest`

## IMPORTANT ARCHITECTURE DECISIONS

### Major Decisions

- Keep modular tool-based architecture.
- Keep `agent.py` as low-level application API.
- Use `core.agent.GOFOAgent` as production/session wrapper.
- Use `core.session.SessionManager` for FastAPI session memory.
- Keep routing in `tools/router.py`.
- Keep planning in `tools/planner/`.
- Keep SQL planning separate from SQL execution.
- Keep RAG retrieval separate from RAG generation.
- Keep ChromaDB access inside `tools/rag/retriever.py`.
- Keep SQLite execution inside `tools/sql/executor.py`.
- Keep LLM initialization inside `tools/llm/client.py`.
- Use SQLite only for current operational and memory databases.
- Use ChromaDB only for SOP vector retrieval.
- Use `drivers.hub` for hub/warehouse/station.
- Use database latest pickup date as operational today.
- Current SQL/RAG results are source of truth; long-term memory is supporting context only.
- `ConversationMemory` handles entities and result context.
- `ConversationState` handles corrections and intent-level repair.
- Streamlit frontend should call FastAPI only and not import agent internals.
- FastAPI must reuse session agents and must not create a new `GOFOAgent` per request.

### What Not To Change Without Explicit Request

DO NOT:

- Recreate working modules.
- Rewrite existing architecture.
- Delete working files or data.
- Create duplicate functionality.
- Move SQL execution out of `tools/sql/executor.py`.
- Move Chroma access out of `tools/rag/retriever.py`.
- Instantiate `ChatOpenAI` outside `tools/llm/client.py`.
- Replace SQLite with PostgreSQL/Snowflake unless explicitly requested.
- Add Redis unless explicitly requested.
- Add LangGraph unless explicitly requested.
- Add multi-agent architecture unless explicitly requested.
- Add external Slack/Lark/Feishu actions unless explicitly requested.
- Change RAG behavior while fixing memory/API bugs.
- Change SQL planner while fixing UI/API bugs.
- Change Streamlit UI while fixing backend memory unless explicitly requested.

### Security and Secrets

- `.env` contains local secrets and must not be committed.
- `.env.example` should contain placeholders only.
- If a real OpenAI key has ever been placed in `.env.example` or logs, rotate it.
- Do not log API keys or sensitive environment variables.

## ENVIRONMENT SETUP

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create `.env` from `.env.example` and fill in local values:

```bash
cp .env.example .env
```

Required:

```text
OPENAI_API_KEY=
```

Common settings:

```text
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
SQLITE_DATABASE=data/gofo_demo.db
MEMORY_DATABASE=data/memory.db
CHROMA_PERSIST_DIR=chroma_db
COLLECTION_NAME=gofo_sop
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
LOG_FILE=logs/agent.log
```

### Database

Create/reset demo analytics database:

```bash
python scripts/create_demo_db.py
python scripts/test_queries.py
```

Persistent memory database initializes automatically on first use:

```text
data/memory.db
```

### Ingest SOP PDFs

```bash
python ingest.py
```

### Run CLI

```bash
python query.py
```

Debug mode:

```bash
python query.py --debug
```

### Run Backend

```bash
uvicorn api.server:app --reload
```

Health:

```bash
curl http://localhost:8000/health
```

Ask:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How are operations today?","session_id":"default"}'
```

### Run Frontend

Main dashboard:

```bash
PYTHONPATH=. streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

Legacy UI:

```bash
streamlit run ui/app.py
```

### Docker

Build:

```bash
docker compose build
```

Run API and UI:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up --build -d
```

Check services:

```bash
docker compose ps
curl http://localhost:8000/health
```

Stop:

```bash
docker compose down
```

### Testing

Run all tests:

```bash
.venv/bin/pytest tests/ -q
```

Recent validation:

```text
179 passed
```

## CURRENT STATUS

### DONE

- PDF ingestion into ChromaDB.
- RAG retriever.
- RAG query rewriter.
- RAG generator.
- RAG confidence filtering.
- RAG service orchestration.
- Shared LLM client.
- Low-level `agent.ask()`.
- Production `GOFOAgent`.
- Natural-language planner.
- SQL planner with schema awareness.
- Live SQLite schema loader.
- SQLite executor.
- SQL summarizer.
- SQL business date resolver.
- SQL service.
- Demo SQLite analytics database.
- Router for SQL/RAG/multi/unknown.
- Multi-tool synthesizer.
- KPI dictionary.
- Operations overview.
- Root-cause analyzer.
- Automatic drilldown for why questions.
- Anomaly detector.
- Recommendation engine.
- Short-term conversation memory.
- Entity memory.
- Result context memory.
- Conversation repair detector.
- Structured `ConversationState`.
- Semantic `ConversationResolver`.
- Context-aware intent classifier.
- Context builder for follow-up reconstruction and stale-filter prevention.
- Evidence-based business reasoner.
- SQL cache support for identical semantic requests.
- FastAPI session memory through `SessionManager`.
- SQLite long-term memory.
- Learned pattern storage.
- Streamlit dashboard.
- CLI and debug REPL.
- API schemas.
- Production logging.
- Centralized error handling.
- FastAPI middleware logging.
- Dockerfile.
- Docker Compose.
- Broad test coverage.
- End-to-end file and image upload intelligence pipeline:
  - `POST /attachments` multipart upload API with validation and persistent storage (`data/uploads/`)
  - `tools/files/` processors for CSV, Excel, PDF, DOCX, TXT/MD, and images (multimodal vision)
  - `tools/context/file_context_builder.py` for compact attachment context
  - `tools/files/source_router.py` for SQLITE / RAG / ATTACHMENT / combined routing
  - `tools/files/attachment_memory.py` for conversational file reference resolution
  - `tools/files/analyzer.py` for file analysis, multi-file comparison, database comparison, and RAG comparison
  - Streamlit `+` attachment button with chips, upload-on-select, and chat history display
  - Configurable upload limits via `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `ALLOWED_UPLOAD_TYPES`
  - `tests/test_attachments.py` covering processors, API, memory, and required conversation flows

### IN PROGRESS

- Real-world prompt quality and SQL accuracy should continue to be monitored.
- Long-term memory retrieval is keyword/entity based, not vector based.
- Conversational reasoning is now semantic-state based, but additional real-world phrasing should be added through generalized state/context improvements rather than one-off bug patches.
- `ConversationMemory` and `ConversationState` remain for compatibility while `ConversationResolver` is the new semantic source of truth.

### NEXT DEVELOPMENT TASKS

Recommended next work:

1. Run longer manual end-to-end smoke tests in the Streamlit dashboard:
   - `Highest performing driver`
   - `Which hub does he belong to?`
   - `Actually I mean lowest driver`
   - `Rank all hubs by performance`
   - `Why is the worst hub performing badly?`
   - `What should operations do?`
   - `Compare yesterday`
   - `Show details`
   - `Actually compare last week`
   - `What is pickup SOP?`
   - `How are operations today?`

2. Add more deterministic tests for real API responses using mocked `ask_core` for:
   - session isolation between different `session_id` values,
   - multi-turn recommendation/explanation chains,
   - result-table drilldowns through API,
   - SQL cache reuse.

3. Expand semantic conversation resolution carefully for more patterns:
   - date corrections,
   - metric corrections,
   - dimension corrections with active filters.

4. Improve long-term memory scoring:
   - better entity extraction for hub/driver/customer,
   - deduplication of repeated findings,
   - optional retention/cleanup policy.

5. Consider deterministic SQL templates for common KPI/ranking queries if LLM SQL planning quality becomes inconsistent.

6. Rotate any API key that may have previously appeared in `.env.example` or logs. The current `.env.example` is placeholder-only.

7. Keep updating this README after major architecture or behavior changes.

## BUG HISTORY & FIXES

### Conversational Reasoning Was Too Rule-Patch Driven

Problem:

- Direct SQL questions worked, but conversational reasoning degraded across follow-ups.
- Examples included:
  - `Why is the worst hub performing badly?` not resolving from the previous hub ranking.
  - `What should operations do?` not using prior analysis.
  - `Compare yesterday` not inheriting the previous operational topic.
  - `Show details` not reliably using previous result context.
  - Fresh driver rankings risking accidental stale filter inheritance.

Fix:

- Added the production conversational pipeline:
  - `ConversationResolver`
  - `IntentClassifier`
  - `ContextBuilder`
  - `BusinessReasoner`
- Semantic conversation state now stores meaning, not only text.
- Rankings produce `best_entity` and `worst_entity` for later references.
- Contextual recommendations/explanations can answer from previous SQL evidence without new SQL.
- Global ranking questions intentionally avoid inherited filters unless the user explicitly names the filter.
- CLI/API debug payloads now include intent, inherited context, business findings, memory update status, and SQL cache hit status.

Files changed:

- `tools/conversation/resolver.py`
- `tools/conversation/__init__.py`
- `tools/planner/intent_classifier.py`
- `tools/context/context_builder.py`
- `tools/context/__init__.py`
- `tools/analyzer/business_reasoner.py`
- `tools/analyzer/__init__.py`
- `core/agent.py`
- `core/models.py`
- `cli/display.py`
- `tests/test_conversational_pipeline.py`

### SQL Planner Did Not Know Schema

Problem:

- Planner generated incorrect SQL and hallucinated tables/columns.

Fix:

- Added schema-aware prompt content.
- Added `tools/sql/schema_loader.py`.
- Reinforced table relationships and known columns.
- Added unknown table guard.

Files changed:

- `tools/sql/planner.py`
- `tools/sql/schema_loader.py`
- `tests/test_sql_planner_schema.py`

### Hub Was Confused With Customer

Problem:

- Hub/warehouse questions could be mapped incorrectly.

Fix:

- Defined hub as operational warehouse/station.
- Always use `drivers.hub`.

Files changed:

- `tools/sql/planner.py`
- SQL planner tests.

### Agent Had Metrics But No Business Reasoning

Problem:

- Agent answered metrics but did not explain root causes, KPI meaning, anomalies, or recommendations.

Fix:

- Added `tools/analysis/`.
- Added KPI dictionary, root-cause analyzer, drilldown, anomaly detector, recommender.

Files changed:

- `tools/analysis/*`
- `config/kpi_dictionary.yaml`
- `tools/router.py`
- related tests.

### No Persistent Long-Term Memory

Problem:

- Memory disappeared after process restart.

Fix:

- Added SQLite `data/memory.db`.
- Added long-term memory modules and learned pattern logic.

Files changed:

- `tools/memory/database.py`
- `tools/memory/long_memory.py`
- `tools/memory/learning.py`
- `tools/memory/findings.py`
- `tests/test_long_memory.py`

### FastAPI Memory Was Stateless

Problem:

- API/Streamlit did not preserve conversation across requests.

Fix:

- Added `core/session.py`.
- FastAPI now maps `session_id` to a persistent `GOFOAgent`.
- Streamlit sends `st.session_state.session_id`.

Files changed:

- `core/session.py`
- `api/server.py`
- `api/schemas.py`
- `frontend/app.py`
- `tests/test_api_memory.py`

### Correction Regression After Docker Migration

Problem:

- Entity memory worked, but corrections reused old entity context.
- Example: after highest driver = Drew Nguyen, `Actually I mean lowest driver` still talked about Drew.

Root cause:

- Entity memory existed, but intent/sort-direction state was missing.

Fix:

- Added `ConversationState`.
- State repair runs before entity memory resolver.
- Corrections modify intent/dimension/sort direction and do not reuse old entity answer.

Files changed:

- `tools/memory/state.py`
- `core/agent.py`
- `api/server.py`
- `tools/memory/__init__.py`
- `tests/test_state_memory.py`

### Production Infrastructure Missing

Problem:

- No environment example, structured logging, centralized errors, Docker, or Compose.

Fix:

- Added production infrastructure.

Files changed:

- `.env.example`
- `core/logger.py`
- `core/errors.py`
- `core/error_handler.py`
- `api/server.py`
- `query.py`
- `cli/repl.py`
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `tests/test_config.py`
- `tests/test_logging.py`
- `tests/test_health_api.py`

### Docker Compose Build Sandbox Issue

Problem:

- Docker build initially failed in sandbox because Docker needed write access under `~/.docker`.

Fix:

- Retried with full permissions.
- Docker build and `docker compose up --build -d` succeeded.

Latest checks:

- API health returned running.
- Streamlit returned HTTP 200.

## FUTURE ROADMAP

Potential future features:

- Better deterministic SQL templates for common operations KPIs.
- More robust SQL validation and query safety.
- More comprehensive state repair rules.
- Long-term memory deduplication and cleanup.
- Persistent user/team profiles.
- Better observability and metrics beyond file logging.
- Production auth.
- Deployment configuration for cloud hosting.
- Snowflake executor for real operational analytics.
- LangGraph workflow orchestration.
- MCP tool integrations.
- Feishu/Lark automation.
- WeChat integration.
- Slack/mobile/React clients.
- Operations automation actions:
  - alerts,
  - escalation recommendations,
  - daily summaries,
  - exception triage.
- Optional vector memory if explicitly requested.
- Multi-agent architecture only if explicitly requested.

## INSTRUCTIONS FOR FUTURE CURSOR AGENTS

Before modifying code:

1. Read this README.md completely.
2. Inspect the current repository.
3. Understand existing architecture.
4. Continue from NEXT DEVELOPMENT TASKS.
5. Do not recreate completed work.
6. Preserve existing design patterns.
7. Make incremental improvements only.
8. Update README.md after every major change.

This README replaces previous Cursor conversation memory.

Additional project-specific rules:

- Preserve the modular tool architecture.
- Keep API/UI thin.
- Keep SQL, RAG, memory, and analysis responsibilities separate.
- Do not introduce Redis, LangGraph, multi-agent architecture, external services, or new databases unless explicitly requested.
- Never log or document real API keys.
- Run tests after substantive changes.
