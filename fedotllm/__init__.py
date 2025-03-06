import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, List, Optional
from pathlib import Path

import typer
from omegaconf import OmegaConf
from rich import print as rprint

from .constants import DEFAULT_QUALITY, PRESETS
from .utils import load_config
from .task import TabularPredictionTask
from .assistant import TabularPredictionAssistant

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


@dataclass
class TimingContext:
    start_time: float
    total_time_limit: float

    @property
    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def time_remaining(self) -> float:
        return self.total_time_limit - self.time_elapsed


@contextmanager
def time_block(description: str, timer: TimingContext):
    """Context manager for timing code blocks and logging the duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logging.info(
            f"It took {duration:.2f} seconds {description}. "
            f"Time remaining: {timer.time_remaining:.2f}/{timer.total_time_limit:.2f}"
        )


def run_assistant(
    task_path: Annotated[
        str, typer.Argument(help="Directory where task files are included")
    ],
    presets: Annotated[
        Optional[str],
        typer.Option("--presets", "-p", help="Presets"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path", "-c", help="Path to the configuration file (config.yaml)"
        ),
    ] = None,
    config_overrides: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config_overrides",
            "-o",
            help="Override config values. Format: key=value or key.nested=value. Can be used multiple times.",
        ),
    ] = None,
    output_filename: Annotated[Optional[str], typer.Option(help="Output File")] = "",
) -> str:
    start_time = time.time()

    logging.info("Starting FedotLLM")

    if presets is None or presets not in PRESETS:
        logging.info(f"Presets is not provided or invalid: {presets}")
        presets = DEFAULT_QUALITY
        logging.info(f"Using default presets: {presets}")
    logging.info(f"Presets: {presets}")

    # Load config with all overrides
    try:
        config = load_config(presets, config_path, config_overrides)
        logging.info("Successfully loaded config")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    timer = TimingContext(start_time=start_time, total_time_limit=config.time_limit)
    with time_block("initializing components", timer):
        rprint("🤖 [bold red] Welcome to FEDOT.LLM [/bold red]")

        rprint("Will use task config:")
        rprint(OmegaConf.to_container(config))
        
        task_path = Path(task_path).resolve()
        assert task_path.is_dir(), "Task path does not exist, please provide a valid directory."
        rprint(f"Task path: {task_path}")
        
        task = TabularPredictionTask.from_path(task_path)
        
        rprint("[green]Task loaded![/green]")
        rprint(task)
        
        assistant = TabularPredictionAssistant(config)
        
    with time_block("preprocessing task", timer):
        task = assistant.preprocess_task(task)
        
        


def main():
    app = typer.Typer()
    app.command()(run_assistant)
    app()


if __name__ == "__main__":
    main()
