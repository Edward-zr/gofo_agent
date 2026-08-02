"""FastAPI server exposing the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import sqlite3
from time import perf_counter

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

import config
from api.schemas import AskRequest, AskResponse, AttachmentInfo, AttachmentUploadResponse, HealthResponse
from core.error_handler import error_payload, handle_error
from core.errors import AgentError, DatabaseError, FileError
from core.agent import GOFOAgent
from core.logger import get_logger
from core.session import SessionManager
from tools.files.service import AttachmentService
from tools.memory.database import get_connection

logger = get_logger("api")

app = FastAPI(
    title="Operations Intelligence Agent",
    description="HTTP API for the GOFO Operations Intelligence Dashboard.",
    version="1.0.0",
)

attachment_service = AttachmentService()


def _create_session_agent() -> GOFOAgent:
    """Create a session agent that shares the API attachment store."""
    import config

    if config.LANGGRAPH_ENABLED:
        from graph.runtime import LangGraphGOFOAgent

        session_agent = LangGraphGOFOAgent(GOFOAgent(), thread_id="pending")
        session_agent.attachment_service = attachment_service
        return session_agent  # type: ignore[return-value]

    session_agent = GOFOAgent()
    session_agent.attachment_service = attachment_service
    return session_agent


# SessionManager needs thread_id per session — wrap factory.
class _LangGraphAwareSessionManager(SessionManager):
    """Session manager that sets LangGraph thread_id to the API session_id."""

    def get_session(self, session_id: str | None = None):
        normalized = (session_id or "default").strip() or "default"
        with self._lock:
            if normalized not in self._sessions:
                import config

                agent = GOFOAgent()
                agent.attachment_service = attachment_service
                if config.LANGGRAPH_ENABLED:
                    from graph.runtime import LangGraphGOFOAgent

                    wrapped = LangGraphGOFOAgent(agent, thread_id=normalized)
                    wrapped.attachment_service = attachment_service
                    self._sessions[normalized] = wrapped  # type: ignore[assignment]
                else:
                    self._sessions[normalized] = agent
            return self._sessions[normalized]


session_manager = _LangGraphAwareSessionManager(agent_factory=_create_session_agent)
agent = session_manager.get_session("default")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and execution time for each request."""
    started = perf_counter()
    response = await call_next(request)
    elapsed_ms = int((perf_counter() - started) * 1000)
    logger.info(
        "%s %s %s %sms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(FileError)
async def file_error_handler(_request: Request, exc: FileError) -> JSONResponse:
    """Return safe JSON for file upload and processing errors."""
    logger.exception("FileError: %s", handle_error(exc))
    return JSONResponse(status_code=400, content=error_payload(exc))


@app.exception_handler(AgentError)
async def agent_error_handler(_request: Request, exc: AgentError) -> JSONResponse:
    """Return safe JSON for known agent errors."""
    logger.exception("AgentError: %s", handle_error(exc))
    status_code = 503 if isinstance(exc, DatabaseError) else 500
    return JSONResponse(status_code=status_code, content=error_payload(exc))


@app.exception_handler(sqlite3.Error)
async def sqlite_error_handler(_request: Request, exc: sqlite3.Error) -> JSONResponse:
    """Return safe JSON for SQLite errors."""
    logger.exception("SQLite error: %s", handle_error(exc))
    return JSONResponse(status_code=500, content=error_payload(exc))


@app.exception_handler(Exception)
async def generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return safe JSON for unexpected errors."""
    logger.exception("Unhandled error: %s", handle_error(exc))
    return JSONResponse(status_code=500, content=error_payload(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service dependency health."""
    return HealthResponse(
        status="running",
        version="1.0",
        database="connected" if _database_connected() else "missing",
        memory="enabled" if _memory_enabled() else "disabled",
        rag="enabled" if config.CHROMA_PERSIST_DIR.exists() else "disabled",
    )


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """Compatibility health endpoint."""
    return health()


@app.post("/attachments", response_model=AttachmentUploadResponse)
async def upload_attachments(
    files: list[UploadFile] = File(...),
    session_id: str = Form("default"),
) -> AttachmentUploadResponse:
    """Upload one or more files and return attachment metadata."""
    uploaded: list[AttachmentInfo] = []
    for upload in files:
        content = await upload.read()
        metadata = attachment_service.upload(
            filename=upload.filename or "upload.bin",
            content=content,
            content_type=upload.content_type,
            conversation_id=session_id,
        )
        uploaded.append(AttachmentInfo.model_validate(metadata.model_dump()))
    return AttachmentUploadResponse(attachments=uploaded)


@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest) -> AskResponse:
    """Ask the agent a question and return dashboard-friendly structured output."""
    try:
        session_agent = session_manager.get_session(body.session_id)
        response = session_agent.ask(body.question, attachment_ids=body.attachments)
        _log_memory_debug(body.session_id, response)
        return AskResponse.model_validate(response)
    except Exception as exc:
        if isinstance(exc, ValueError):
            return JSONResponse(status_code=422, content=error_payload(exc))
        raise


def _database_connected() -> bool:
    if not config.SQLITE_DATABASE.exists():
        return False
    try:
        connection = sqlite3.connect(str(config.SQLITE_DATABASE))
        connection.execute("SELECT 1;")
        connection.close()
        return True
    except sqlite3.Error:
        return False


def _memory_enabled() -> bool:
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1 FROM conversation_history LIMIT 1;")
        return True
    except sqlite3.Error:
        return False


def _log_memory_debug(session_id: str, response: dict) -> None:
    """Log session memory state for debugging API conversation continuity."""
    analysis = response.get("analysis") or {}
    logger.info("SESSION ID: %s", session_id)
    logger.info("Memory turns: %s", analysis.get("memory_turns"))
    logger.info("Previous question: %s", analysis.get("previous_question"))
    logger.info("Previous answer: %s", analysis.get("previous_answer"))
    logger.info("Current entities: %s", analysis.get("current_entities"))
    logger.info("Repair detected: %s", analysis.get("repair_detected"))
    logger.info("Previous state: %s", analysis.get("previous_state"))
    logger.info("New state: %s", analysis.get("new_state"))
    logger.info("Resolved question: %s", analysis.get("resolved_question"))
    logger.info("Attachment IDs: %s", analysis.get("attachment_ids"))
    logger.info("Data sources: %s", analysis.get("data_sources"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host=config.API_HOST, port=config.API_PORT, reload=True)
