"""
FedotLLM Server Application

FastAPI server that provides REST API endpoints for the FedotLLM package.
This serves as the backend for the frontend application and provides
programmatic access to FedotLLM functionality.
"""

from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
tasks: Dict[str, Dict] = {}


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
    background_tasks: BackgroundTasks,
):
    """
    Create and start a new AutoML task.

    Args:
        request: Task request containing task directory and configuration
        background_tasks: FastAPI background tasks

    Returns:
        TaskResponse with task ID and initial status
    """
    task_id = f"task_{len(tasks) + 1}"

    # Validate task directory
    task_dir = Path(request.task_dir)
    if not task_dir.exists():
        raise HTTPException(status_code=400, detail=f"Task directory not found: {request.task_dir}")

    # Initialize task
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "logs": "",
        "output_file": None,
        "request": request,
    }

    # Start task in background
    background_tasks.add_task(run_task_background, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status="pending",
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

    # Cancel task (implement proper cancellation logic)
    tasks[task_id]["status"] = "cancelled"

    return {"message": f"Task {task_id} cancelled"}


def run_task_background(task_id: str, request: TaskRequest) -> None:
    """
    Run task in the background.

    Args:
        task_id: The task identifier
        request: Task request with configuration
    """
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 10

        task_path = Path(request.task_dir).resolve()
        output_filename = str(task_path / f"fedotllm_output_{task_id}.csv")

        run_assistant(
            str(task_path),
            presets=request.presets,
            config_overrides=request.config_overrides,
            output_filename=output_filename,
        )

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["output_file"] = output_filename
        tasks[task_id]["logs"] = "Task completed successfully"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["logs"] = str(e)


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
