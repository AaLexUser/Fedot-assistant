"""FastAPI server that exposes FedotLLM task execution APIs."""

from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fedotllm import run_assistant
from fedotllm.constants import PRESETS, PROBLEM_TYPES, TASK_TYPES
from pydantic import BaseModel
from shared import AUTOML_ENGINE_OPTIONS, TIME_LIMIT_MAPPING


# Request/Response Models
class TaskRequest(BaseModel):
    task_dir: str
    config_overrides: Optional[List[str]] = None
    presets: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    logs: Optional[str] = None
    output_file: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigOptionsResponse(BaseModel):
    presets: List[str]
    problem_types: List[str]
    task_types: List[str]
    time_limits: Dict[str, int]
    automl_engines: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="FedotLLM API",
    description="LLM-based multi-AutoML Orchestrator API",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory task storage (use proper database in production)
tasks: Dict[str, Dict[str, Any]] = {}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/config/options", response_model=ConfigOptionsResponse)
async def get_config_options():
    """Get available configuration options."""
    return ConfigOptionsResponse(
        presets=PRESETS,
        problem_types=PROBLEM_TYPES,
        task_types=TASK_TYPES,
        time_limits=TIME_LIMIT_MAPPING,
        automl_engines=AUTOML_ENGINE_OPTIONS,
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
):
    """
    Create and start a new AutoML task.

    Args:
        request: Task request containing task directory and configuration
    Returns:
        TaskResponse with task ID and initial status
    """
    task_id = f"task_{len(tasks) + 1}"

    # Validate task directory
    task_dir = Path(request.task_dir)
    if not task_dir.exists():
        raise HTTPException(status_code=400, detail=f"Task directory not found: {request.task_dir}")

    output_filename = str(task_dir.resolve() / f"fedotllm_output_{task_id}.csv")
    process = Process(
        target=run_task_background,
        args=(request, output_filename),
    )
    process.start()

    tasks[task_id] = {
        "status": "running",
        "progress": 10,
        "logs": "Task is running",
        "output_file": None,
        "output_path": output_filename,
        "pid": process.pid,
        "process": process,
        "request": request,
    }

    return TaskResponse(
        task_id=task_id,
        status="running",
        message="Task created successfully",
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a running task.

    Args:
        task_id: The task identifier

    Returns:
        TaskStatusResponse with current task status
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    _refresh_task_state(task_id)
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        logs=task.get("logs"),
        output_file=task.get("output_file"),
    )


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running task.

    Args:
        task_id: The task identifier

    Returns:
        Success message
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    _refresh_task_state(task_id)
    task = tasks[task_id]
    if task["status"] != "running":
        return {"message": f"Task {task_id} is already {task['status']}"}

    process = _get_task_process(task)
    if process is None or not process.is_alive():
        _refresh_task_state(task_id)
        return {"message": f"Task {task_id} is already {tasks[task_id]['status']}"}

    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)

    output_path = Path(cast(str, task["output_path"]))
    if output_path.exists():
        output_path.unlink()

    task["status"] = "cancelled"
    task["logs"] = "Task cancelled"
    task["progress"] = 0
    task["process"] = None
    task["pid"] = None
    task["output_file"] = None

    return {"message": f"Task {task_id} cancelled"}


def _get_task_process(task: Dict[str, Any]) -> Optional[Process]:
    process = task.get("process")
    if isinstance(process, Process):
        return process
    return None


def _refresh_task_state(task_id: str) -> None:
    task = tasks[task_id]
    if task["status"] != "running":
        return

    process = _get_task_process(task)
    if process is None or process.is_alive():
        return

    process.join(timeout=0)
    task["process"] = None
    task["pid"] = None

    if process.exitcode == 0:
        task["status"] = "completed"
        task["progress"] = 100
        task["output_file"] = cast(str, task["output_path"])
        task["logs"] = "Task completed successfully"
        return

    task["status"] = "failed"
    task["logs"] = f"Task failed with exit code {process.exitcode}"


def run_task_background(request: TaskRequest, output_filename: str) -> None:
    """Run the requested task in a dedicated worker process."""
    task_path = Path(request.task_dir).resolve()
    run_assistant(
        str(task_path),
        presets=request.presets,
        config_overrides=request.config_overrides,
        output_filename=output_filename,
    )


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
