import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, List, Optional, Union
from pathlib import Path
import pandas as pd
import datetime

import typer
from omegaconf import OmegaConf
from rich import print as rprint

from .constants import DEFAULT_QUALITY, PRESETS, NO_ID_COLUMN_IDENTIFIED
from .utils import load_config
from .task import PredictionTask
from .assistant import PredictionAssistant

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


def make_prediction_outputs(
    task: PredictionTask, predictions: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    if isinstance(predictions, pd.Series):
        outputs = predictions.to_frame()
    else:
        outputs = predictions.copy()

    # Ensure we only keep required output columns from predictions
    common_cols = [col for col in task.output_columns if col in outputs.columns]
    outputs = outputs[common_cols]

    # Handle specific test ID column if providded and detected
    if (
        task.test_id_column is not None
        and task.test_id_column != NO_ID_COLUMN_IDENTIFIED
    ):
        test_ids = task.test_data[task.test_id_column]
        output_ids = task.sample_submission_data[task.output_id_column]

        if not test_ids.equals(output_ids):
            print("Warming: Test IDs and output IDs do not match!")

        # Ensure test ID column is included
        if task.test_id_column not in outputs.columns:
            outputs = pd.concat(
                [task.test_data[task.test_id_column], outputs], axis="columns"
            )

    # Handle undetected ID columns
    missing_columns = [col for col in task.output_columns if col not in outputs.columns]
    if missing_columns:
        print(
            "Warming: The following columns are not in predictions and will be treated as ID columns:"
            f"{missing_columns}"
        )

        for col in missing_columns:
            if task.test_data is not None and col in task.test_data.columns:
                # Copy from test data if available
                outputs[col] = task.test_data[col]
                print(f"Warming: Copied from test data for column '{col}'")
            else:
                # Generate unique integer values
                outputs[col] = range(len(outputs))
                print(
                    f"Warming: Generated unique integer values for column '{col}'"
                    "as it was not found in test data"
                )

    # Ensure columns are in the correct order
    outputs = outputs[task.output_columns]

    return outputs


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
        assert task_path.is_dir(), (
            "Task path does not exist, please provide a valid directory."
        )
        rprint(f"Task path: {task_path}")

        task = PredictionTask.from_path(task_path)

        rprint("[green]Task loaded![/green]")
        rprint(task)

        assistant = PredictionAssistant(config)

    with time_block("preprocessing task", timer):
        task = assistant.preprocess_task(task)

    with time_block("training model", timer):
        rprint("Model training starts...")

        assistant.fit_predictor(task, time_limit=timer.time_remaining)

        rprint("[green]Model training complete![/green]")

    with time_block("making predictions", timer):
        rprint("Prediction starts...")

        predictions = assistant.predict(task)

        if not output_filename:
            output_filename = (
                f"fedotllm-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        with open(output_filename, "w") as fp:
            make_prediction_outputs(task, predictions).to_csv(fp, index=False)

        rprint(
            f"[green] Prediction complete! Outputs written to {output_filename}[/green]"
        )

    if config.save_artifacts.enabled:
        artifacts_dir_name = f"{task.metadata['name']}_artifacts"
        if config.save_artifacts.append_timestamp:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir_name = (
                f"{task.metadata['name']}_artifacts_{current_timestamp}"
            )

        full_save_path = Path(config.save_artifacts.path) / artifacts_dir_name

        task.save_artifacts(full_save_path, assistant.predictor)

        rprint(
            f"[green]Artifacts including transformed datasets and trained model saved at {full_save_path}"
        )

    return output_filename


def main():
    app = typer.Typer()
    app.command()(run_assistant)
    app()


if __name__ == "__main__":
    main()
