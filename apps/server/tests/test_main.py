import asyncio
from pathlib import Path

import pytest
from server import main


class FakeProcess:
    next_pid = 1000

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.started = False
        self.alive = False
        self.exitcode = None
        self.terminated = False
        self.killed = False

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self.alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


@pytest.fixture(autouse=True)
def clear_tasks():
    main.tasks.clear()
    yield
    main.tasks.clear()


def test_cancel_task_terminates_worker_and_removes_output(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "Process", FakeProcess)

    request = main.TaskRequest(task_dir=str(tmp_path))
    response = asyncio.run(main.create_task(request))
    task_id = response.task_id

    output_path = Path(main.tasks[task_id]["output_path"])
    output_path.write_text("partial", encoding="utf-8")

    result = asyncio.run(main.cancel_task(task_id))

    assert result == {"message": f"Task {task_id} cancelled"}
    assert main.tasks[task_id]["status"] == "cancelled"
    assert main.tasks[task_id]["process"] is None
    assert main.tasks[task_id]["pid"] is None
    assert main.tasks[task_id]["output_file"] is None
    assert not output_path.exists()


def test_get_task_status_refreshes_completed_process(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "Process", FakeProcess)

    request = main.TaskRequest(task_dir=str(tmp_path))
    response = asyncio.run(main.create_task(request))
    task_id = response.task_id
    task = main.tasks[task_id]
    process = task["process"]
    assert isinstance(process, FakeProcess)

    process.alive = False
    process.exitcode = 0

    status = asyncio.run(main.get_task_status(task_id))

    assert status.status == "completed"
    assert status.output_file == task["output_path"]
    assert main.tasks[task_id]["process"] is None
    assert main.tasks[task_id]["pid"] is None
