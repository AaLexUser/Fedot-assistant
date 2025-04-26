# app.py
import asyncio
import logging
import re
import uuid
import shlex  # For safer command construction
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl, SecretStr, Field

# Use absolute imports if project structure allows, otherwise adjust relative paths
# Assume fedotllm is installed or in PYTHONPATH
from fedotllm.constants import PRESETS, DEFAULT_QUALITY

# Import from our refactored event/stream modules
from .stream import EventStream, EventStreamSubscriber
from .event import (
    Event,
    LogEvent,
    ErrorEvent,
    StatusEvent,
    ResultEvent,
    CompleteEvent,
)

# --- Constants ---
SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
SSE_TIMEOUT = 15.0
DEFAULT_RESULT_FILENAME = "submission.csv"  # Predictable result filename

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Reduce verbosity of noisy libraries
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Request Models ---
class SessionInitRequest(BaseModel):
    """User-provided parameters to start processing."""

    model: Optional[str] = Field(None, description="LLM model name to use")
    api_key: Optional[SecretStr] = Field(None, description="API key for the LLM")
    base_url: Optional[HttpUrl] = Field(
        None, description="Optional base URL for LLM API"
    )
    description: str = Field("", description="Initial prompt/description for the agent")
    preset: Optional[str] = Field(
        DEFAULT_QUALITY,
        description=f"Preset configuration name ({', '.join(PRESETS)})",
    )
    config_overrides: Optional[List[str]] = Field(
        None,
        description="Advanced: Override specific config keys (e.g., 'llm.temperature=0.5')",
    )  # Keep it


# --- Session Class ---
class Session:
    """Represents a user session, managing state, files, and process execution."""

    def __init__(self, session_id: str):
        self.id: str = session_id
        self.dir: Path = SESSIONS_DIR / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.stream: EventStream = EventStream(session_id)
        self.sse_queue: asyncio.Queue[Optional[Event]] = asyncio.Queue()

        self.process: Optional[asyncio.subprocess.Process] = None
        self.process_task: Optional[asyncio.Task] = None
        self.log_reader_task: Optional[asyncio.Task] = None

        self.active_processing: bool = False
        self.active_sse_connection: bool = False
        self.result_filename: Optional[str] = (
            None  # Store the name of the generated result file
        )

        self._file_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()  # Lock for modifying processing state flags

        logger.info(f"Session {self.id} initialized. Directory: {self.dir}")
        # Start stream processing immediately after initialization
        asyncio.create_task(self._initialize_stream_processing())

    async def _initialize_stream_processing(self):
        """Starts the event stream and subscribes the SSE queue."""
        await self.stream.start_processing()
        self.stream.subscribe(
            subscriber_id=EventStreamSubscriber.SERVER_SSE,
            callback=self._sse_event_handler,
            callback_id=f"sse_forwarder_{self.id}",
        )
        logger.info(f"SSE queue subscribed to event stream for session {self.id}")

    async def _sse_event_handler(self, event: Event):
        """Puts events from the EventStream onto the SSE queue."""
        await self.sse_queue.put(event)

    async def set_processing_status(self, active: bool):
        """Safely set the processing status."""
        async with self._state_lock:
            self.active_processing = active

    async def is_processing(self) -> bool:
        """Safely check the processing status."""
        async with self._state_lock:
            return self.active_processing

    async def cleanup(self):
        """Clean up session resources: stop process, stream, tasks."""
        logger.info(f"Initiating cleanup for session {self.id}")

        async with self._state_lock:
            self.active_processing = False
            self.active_sse_connection = False

        # 1. Terminate subprocess if running
        if self.process and self.process.returncode is None:
            logger.warning(
                f"Force terminating assistant process PID {self.process.pid} for session {self.id}"
            )
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error(
                    f"Process {self.process.pid} did not terminate gracefully, killing."
                )
                self.process.kill()
            except ProcessLookupError:
                logger.warning(f"Process {self.process.pid} already finished.")
            except Exception as e:
                logger.error(
                    f"Error terminating process {self.process.pid}: {e}", exc_info=True
                )
            self.process = None

        # 2. Cancel background tasks (log reader, main process task)
        for task in [self.log_reader_task, self.process_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task  # Allow cancellation to propagate
                except asyncio.CancelledError:
                    logger.debug(
                        f"Task {task.get_name()} cancelled successfully for session {self.id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error awaiting cancelled task {task.get_name()} for session {self.id}: {e}",
                        exc_info=True,
                    )
        self.log_reader_task = None
        self.process_task = None

        # 3. Unsubscribe and signal SSE queue
        try:
            self.stream.unsubscribe(
                EventStreamSubscriber.SERVER_SSE, f"sse_forwarder_{self.id}"
            )
        except Exception as e:  # Catch specific errors if necessary
            logger.warning(f"Error unsubscribing SSE queue for session {self.id}: {e}")
        await self.sse_queue.put(None)  # Signal SSE generator to stop

        # 4. Close the event stream
        try:
            await self.stream.close()
            logger.info(f"Event stream closed for session {self.id}")
        except Exception as e:
            logger.error(
                f"Error closing event stream for session {self.id}: {e}", exc_info=True
            )

        logger.info(
            f"Session {self.id} cleanup complete. Directory preserved at: {self.dir}"
        )

    async def get_file_lock(self) -> asyncio.Lock:
        """Provides access to the session's file operation lock."""
        return self._file_lock


# --- Global Session Management ---
sessions: Dict[str, Session] = {}
sessions_lock = asyncio.Lock()


# --- Helper Function ---
async def send_event(session: Session, event: Event):
    """Helper to add an event to the session's stream."""
    try:
        await session.stream.add_event(event)
    except Exception as e:
        logger.error(
            f"Failed to send event type {event.type} for session {session.id}: {e}",
            exc_info=True,
        )


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (create dirs) and shutdown (cleanup sessions)."""
    logger.info("Server starting up...")
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sessions directory ensured: {SESSIONS_DIR}")
    # Note: No session state restoration from disk in this simplified version.
    # Existing directories might correspond to orphaned sessions.
    yield  # Application runs here
    logger.info("Server shutting down. Cleaning up active sessions...")
    async with sessions_lock:
        active_session_ids = list(sessions.keys())  # Copy keys
    if active_session_ids:
        logger.info(f"Found {len(active_session_ids)} sessions to clean up.")
        cleanup_tasks = [
            asyncio.create_task(
                cleanup_session(sid, acquire_lock=False)
            )  # Lock held implicitly by caller context
            for sid in active_session_ids
        ]
        await asyncio.gather(
            *cleanup_tasks, return_exceptions=True
        )  # Log errors from cleanup
        logger.info("Session cleanup tasks completed.")
    else:
        logger.info("No active sessions needed cleanup.")
    logger.info("Server shutdown complete.")


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan, title="FedotLLM SSE Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Session Management Functions ---
async def get_session(session_id: str) -> Session:
    """Retrieves a session by ID."""
    if not is_valid_uuid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format.")
    async with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            # Optionally check if session_dir exists and decide if you want to "revive" it
            # For simplicity, we only work with in-memory sessions.
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found.")
    return session


async def cleanup_session(session_id: str, acquire_lock: bool = True):
    """Removes session from memory and cleans up its resources."""
    session_to_clean = None
    if acquire_lock:
        async with sessions_lock:
            session_to_clean = sessions.pop(session_id, None)
    else:  # Assume lock is already held (e.g., during shutdown)
        session_to_clean = sessions.pop(session_id, None)

    if session_to_clean:
        logger.info(f"Starting cleanup for session: {session_id}")
        await session_to_clean.cleanup()
    else:
        logger.warning(
            f"Attempted cleanup, but session {session_id} not found in active sessions."
        )


def is_valid_uuid(uuid_str: str) -> bool:
    """Checks if a string is a valid UUID."""
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False


# --- Subprocess Execution and Log Reading ---


async def read_stream(stream: asyncio.StreamReader, session: Session, level: str):
    """Reads lines from a stream (stdout/stderr) and sends LogEvents."""
    while not stream.at_eof():
        try:
            line_bytes = await stream.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line:
                # Simple check for the final output filename pattern
                # Adjust regex if needed based on actual assistant output format
                match = re.search(r"Outputs written to\s+(.+)", line)
                if match:
                    filename = match.group(1).strip()
                    session.result_filename = Path(
                        filename
                    ).name  # Store only the basename
                    logger.info(
                        f"Detected result file '{session.result_filename}' for session {session.id}"
                    )
                    await send_event(
                        session,
                        ResultEvent(
                            filename=session.result_filename,
                            message=f"Result file generated: {session.result_filename}",
                        ),
                    )

                # Format log line slightly (optional)
                formatted_line = re.sub(
                    r"\x1B\[[0-?]*[ -/]*[@-~]", "", line
                )  # Remove ANSI codes
                log_event = LogEvent(message=formatted_line, level=level.upper())
                await send_event(session, log_event)
        except UnicodeDecodeError as e:
            err_msg = f"Encoding error reading assistant output: {e}"
            logger.error(err_msg)
            await send_event(session, ErrorEvent(message=err_msg, details=str(e)))
        except Exception as e:
            err_msg = f"Error reading assistant stream: {e}"
            logger.exception(f"Session {session.id}: {err_msg}")
            await send_event(session, ErrorEvent(message=err_msg, details=str(e)))
            break  # Stop reading on unexpected errors
    logger.debug(f"Stream reading finished for {level} in session {session.id}")


async def run_assistant_process(session: Session, init_params: SessionInitRequest):
    """Runs the assistant subprocess and monitors its output."""
    await session.set_processing_status(True)
    session.result_filename = None  # Reset result filename

    output_file_path = session.dir / DEFAULT_RESULT_FILENAME
    session.result_filename = DEFAULT_RESULT_FILENAME  # Assume this name initially

    command = [
        "fedotllm",  # Assumes 'fedotllm' command is in PATH
        str(session.dir.resolve()),  # Task path (session directory)
        "--output-filename",
        str(output_file_path.resolve()),  # Explicit output path
    ]

    # Add preset if specified and valid
    if init_params.preset and init_params.preset in PRESETS:
        command.extend(["--presets", init_params.preset])

    # Add description if provided
    if init_params.description:
        command.extend(
            ["--description", shlex.quote(init_params.description)]
        )  # Quote description

    # --- Add Config Overrides ---
    overrides = []
    if init_params.model:
        overrides.append(f"llm.model={shlex.quote(init_params.model)}")
    if init_params.api_key:
        overrides.append(
            f"llm.api_key={shlex.quote(init_params.api_key.get_secret_value())}"
        )
    if init_params.base_url:
        overrides.append(f"llm.base_url={shlex.quote(str(init_params.base_url))}")

    if init_params.config_overrides:
        overrides.extend([shlex.quote(ov) for ov in init_params.config_overrides])

    if overrides:
        command.extend(["--config-overrides", *overrides])

    logger.info(
        f"Session {session.id}: Running command: {' '.join(command)}"
    )  # Log for debugging
    await send_event(session, StatusEvent(message="Starting assistant process..."))

    process = None
    success = False
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # Capture stderr as well
            cwd=SESSIONS_DIR,  # Run from sessions dir parent? Or maybe cwd=session.dir? Check fedotllm needs.
        )
        session.process = process  # Store process handle
        logger.info(
            f"Session {session.id}: Assistant process started with PID {process.pid}"
        )
        await send_event(
            session,
            StatusEvent(message=f"Assistant process started (PID: {process.pid})"),
        )

        # Start concurrent tasks to read stdout and stderr
        stdout_task = asyncio.create_task(
            read_stream(process.stdout, session, "stdout"),
            name=f"stdout-reader-{session.id[:8]}",
        )
        stderr_task = asyncio.create_task(
            read_stream(process.stderr, session, "stderr"),
            name=f"stderr-reader-{session.id[:8]}",
        )
        session.log_reader_task = asyncio.gather(
            stdout_task, stderr_task
        )  # Group log readers

        # Wait for the process to complete
        return_code = await process.wait()
        logger.info(
            f"Session {session.id}: Assistant process {process.pid} finished with exit code {return_code}"
        )

        # Wait for log readers to finish processing remaining output
        await asyncio.wait_for(
            session.log_reader_task, timeout=10.0
        )  # Timeout for log readers

        if return_code == 0:
            # Check if result file exists as final confirmation
            if output_file_path.is_file():
                await send_event(
                    session,
                    StatusEvent(message="Assistant process completed successfully."),
                )
                success = True
            else:
                err_msg = f"Assistant process finished with code 0, but result file '{DEFAULT_RESULT_FILENAME}' not found."
                logger.error(f"Session {session.id}: {err_msg}")
                await send_event(session, ErrorEvent(message=err_msg))
                success = False
        else:
            err_msg = f"Assistant process failed with exit code {return_code}."
            logger.error(f"Session {session.id}: {err_msg}")
            # Send error event, stderr should have been captured by read_stream
            await send_event(session, ErrorEvent(message=err_msg))
            success = False

    except FileNotFoundError:
        err_msg = "Error: 'fedotllm' command not found. Make sure FedotLLM is installed and in the system PATH."
        logger.error(err_msg)
        await send_event(
            session,
            ErrorEvent(message=err_msg, details="Check server environment setup."),
        )
        success = False
    except asyncio.TimeoutError:
        err_msg = "Timeout waiting for log readers to finish after process exit."
        logger.error(f"Session {session.id}: {err_msg}")
        await send_event(session, ErrorEvent(message=err_msg))
        success = False
    except Exception as e:
        err_msg = f"An unexpected error occurred while running the assistant: {type(e).__name__}"
        logger.exception(f"Session {session.id}: Error running assistant process")
        await send_event(session, ErrorEvent(message=err_msg, details=str(e)))
        success = False
    finally:
        # Ensure process is marked as finished and reference is cleared
        session.process = None
        session.log_reader_task = None  # Clear task reference
        await send_event(
            session, CompleteEvent(success=success, message="Task processing finished.")
        )
        await session.set_processing_status(False)
        logger.info(
            f"Session {session.id}: Processing marked as finished (success={success})."
        )


# --- API Endpoints ---


@app.post("/sessions", summary="Create New Session", status_code=201)
async def create_new_session_endpoint():
    """Creates a new session and returns its unique ID."""
    session_id = str(uuid.uuid4())
    session = Session(session_id)
    async with sessions_lock:
        sessions[session_id] = session
    logger.info(f"Created new session: {session_id}")
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/files", summary="Upload Files to Session")
async def upload_files_endpoint(session_id: str, files: List[UploadFile] = File(...)):
    """Uploads files required by the assistant to the session directory."""
    session = await get_session(session_id)
    if await session.is_processing():
        raise HTTPException(
            status_code=409, detail="Cannot upload files while processing is active."
        )

    uploaded_filenames = []
    file_lock = await session.get_file_lock()

    for file in files:
        if not file.filename:
            logger.warning(
                f"Skipping file upload for session {session_id} due to missing filename."
            )
            continue  # Skip files without names

        # Sanitize filename
        safe_filename = Path(file.filename).name  # Use only the basename
        if not safe_filename:
            logger.warning(
                f"Skipping file upload for session {session_id} due to invalid filename '{file.filename}'"
            )
            continue

        target_path = session.dir / safe_filename
        if ".." in str(target_path):  # Double check against manipulation
            raise HTTPException(
                status_code=400, detail=f"Invalid filename: {safe_filename}"
            )

        try:
            async with file_lock:
                async with aiofiles.open(target_path, "wb") as f:
                    # Read file in chunks for potentially large files
                    while content := await file.read(1024 * 1024):  # Read 1MB chunks
                        await f.write(content)
            uploaded_filenames.append(safe_filename)
            logger.info(f"Uploaded file '{safe_filename}' to session {session_id}")
        except Exception as e:
            logger.error(
                f"Failed to save file '{safe_filename}' for session {session_id}: {e}",
                exc_info=True,
            )
            # Attempt to clean up partially written file
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {safe_filename}"
            )
        finally:
            await file.close()

    if uploaded_filenames:
        await send_event(
            session, StatusEvent(message=f"Uploaded {len(uploaded_filenames)} file(s).")
        )
    return {
        "message": f"Uploaded {len(uploaded_filenames)} files.",
        "uploaded_files": uploaded_filenames,
    }


@app.post("/sessions/{session_id}/start", summary="Start Assistant Processing")
async def start_session_processing_endpoint(
    session_id: str, init_params: SessionInitRequest, background_tasks: BackgroundTasks
):
    """Starts the background assistant process for the session."""
    session = await get_session(session_id)
    if await session.is_processing():
        logger.warning(f"Attempted start on already active session: {session_id}")
        raise HTTPException(
            status_code=409, detail="Processing is already active for this session."
        )

    logger.info(f"Adding assistant task to background for session: {session_id}")
    session.process_task = asyncio.create_task(
        run_assistant_process(session, init_params),
        name=f"process-task-{session.id[:8]}",
    )
    # background_tasks.add_task(run_assistant_process, session, init_params) # Using create_task directly for better handle access
    return {"message": "Assistant processing started in background. Monitor events."}


@app.get("/sessions/{session_id}/events", summary="Stream Session Events (SSE)")
async def stream_session_events_endpoint(session_id: str):
    """Establishes an SSE connection to stream real-time events."""
    session = await get_session(session_id)
    if session.active_sse_connection:
        # Reject new connections
        logger.warning(
            f"New SSE connection attempt for session {session_id} while one might be active."
        )
        raise HTTPException(
            status_code=409,
            detail="An SSE connection is already active for this session.",
        )

    session.active_sse_connection = True
    logger.info(f"SSE client connected for session: {session.id}")

    async def event_generator():
        queue = session.sse_queue
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_TIMEOUT)
                    if event is None:  # Sentinel from cleanup
                        logger.info(
                            f"SSE generator received None sentinel for session {session.id}. Stopping."
                        )
                        break

                    try:
                        event_json = event.model_dump_json(
                            exclude={"_id", "_timestamp"}, exclude_none=True
                        )
                        sse_data = f"id: {event.id}\nevent: {event.type.value}\ndata: {event_json}\n\n"
                        yield sse_data
                    except Exception as json_err:
                        logger.error(
                            f"SSE serialization error for session {session.id}: {json_err}",
                            exc_info=True,
                        )
                        # Send a generic error event if serialization fails
                        error_event = ErrorEvent(
                            message="Internal error: Failed to serialize event data."
                        )
                        yield f"event: error\ndata: {error_event.model_dump_json(exclude={'_id', '_timestamp'})}\n\n"
                    finally:
                        queue.task_done()

                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # Keep connection alive
                except asyncio.CancelledError:
                    logger.info(
                        f"SSE event generator cancelled for session {session.id}"
                    )
                    break  # Exit loop cleanly on cancellation
        except Exception as e:
            logger.error(
                f"Unexpected error in SSE generator for session {session.id}: {e}",
                exc_info=True,
            )
            # Attempt to send final error message if possible
            try:
                error_event = ErrorEvent(message="Unexpected error in event stream.")
                yield f"event: error\ndata: {error_event.model_dump_json(exclude={'_id', '_timestamp'})}\n\n"
            except Exception:
                pass  # Avoid further errors
        finally:
            logger.info(f"SSE event generator finished for session {session.id}")
            session.active_sse_connection = False  # Mark connection as closed
            # Ensure queue is marked done if exited unexpectedly with an item
            if "event" in locals() and event is not None and not queue.empty():
                queue.task_done()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Important for Nginx buffering
        },
    )


@app.get("/sessions/{session_id}/download", summary="Download Result File")
async def download_result_file_endpoint(
    session_id: str,
    filename: Optional[str] = Query(
        None, description="Specific filename to download (optional, defaults to result)"
    ),
):
    """Downloads the result file generated by the assistant."""
    session = await get_session(session_id)

    if await session.is_processing():
        raise HTTPException(
            status_code=409,
            detail="Cannot download results while processing is active.",
        )

    # Determine the target filename
    target_filename = filename if filename else session.result_filename
    if not target_filename:
        # Try default name if result_filename wasn't explicitly set/detected
        target_filename = DEFAULT_RESULT_FILENAME
        logger.warning(
            f"Session {session.id}: Result filename not explicitly set, attempting default '{target_filename}'"
        )

    if not target_filename:  # If still no filename
        raise HTTPException(
            status_code=404,
            detail="Result filename unknown or processing did not complete successfully.",
        )

    # Sanitize filename again
    safe_filename = Path(target_filename).name
    if not safe_filename or ".." in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename requested.")

    file_path = (session.dir / safe_filename).resolve()

    # Security check: Ensure path is within session directory
    if not str(file_path).startswith(str(session.dir.resolve())):
        logger.error(
            f"Session {session.id}: Attempt to download file outside session dir: '{safe_filename}' resolved to {file_path}"
        )
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.is_file():
        logger.warning(
            f"Session {session.id}: Requested download file not found: {file_path}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Result file '{safe_filename}' not found. Task might have failed or file name is incorrect.",
        )

    logger.info(f"Serving download for session {session.id}: {safe_filename}")
    return FileResponse(
        path=file_path,
        filename=safe_filename,  # Send the sanitized name to the client
        media_type="application/octet-stream",
    )


@app.delete("/sessions/{session_id}", summary="Delete Session", status_code=200)
async def delete_session_endpoint(session_id: str):
    """Initiates cleanup for a session and removes it from memory."""
    # get_session ensures it exists or raises 404
    await get_session(session_id)
    await cleanup_session(session_id)  # Uses lock internally
    return {"message": f"Session {session_id} cleanup initiated."}


@app.get("/health", summary="Health Check", tags=["Meta"])
async def health_check_endpoint():
    """Basic health check."""
    async with sessions_lock:
        active_session_count = len(sessions)
    return {"status": "ok", "active_sessions": active_session_count}


# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FedotLLM SSE Server with Uvicorn...")
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=7890,
        log_level="info",
        reload=True,  # Enable reload for development, disable in production
        # timeout_keep_alive=65 # Slightly longer than SSE timeout if needed, but 0 might be better for long SSE
    )
