import datetime
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional, Tuple, Union

import pandas as pd
import typer
from omegaconf import OmegaConf
from rich import print as rprint

from .assistant import PredictionAssistant
from .constants import DEFAULT_QUALITY, NO_ID_COLUMN_IDENTIFIED, PRESETS
from .task import PredictionTask
from .utils import load_config

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _coerce_to_reference_dtype(series: pd.Series, reference: pd.Series) -> pd.Series:
    try:
        if pd.api.types.is_datetime64_any_dtype(reference.dtype):
            coerced = pd.to_datetime(series, errors="raise")
        elif pd.api.types.is_integer_dtype(reference.dtype):
            coerced = pd.to_numeric(series, errors="raise")
            non_null = coerced[coerced.notna()]
            if not non_null.empty and not ((non_null % 1) == 0).all():
                raise ValueError("non-integer values cannot be coerced to integer")
            coerced = coerced.astype(reference.dtype)
        elif pd.api.types.is_float_dtype(reference.dtype):
            coerced = pd.to_numeric(series, errors="raise").astype(reference.dtype)
        else:
            coerced = series.astype(reference.dtype)
    except (TypeError, ValueError) as exc:
        print(
            f"WARNING: Could not coerce output column '{series.name}' "
            f"to expected dtype '{reference.dtype}': {exc}"
        )
        return series

    return pd.Series(coerced.to_numpy(), index=series.index, name=series.name)


def make_prediction_outputs(
    task: PredictionTask, predictions: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    if isinstance(predictions, pd.Series):
        outputs = predictions.to_frame()
    else:
        outputs = predictions.copy()

    if task.output_columns is None:
        return outputs

    sample_submission = task.sample_submission_data
    sample_submission_matches = sample_submission is not None and len(
        sample_submission
    ) == len(outputs)

    if sample_submission_matches:
        test_columns = set()
        if task.test_data is not None:
            test_columns.update(task.test_data.columns)
            test_columns.update(name for name in task.test_data.index.names if name)

        prediction_columns = [
            col
            for col in task.output_columns
            if col != task.output_id_column and col not in test_columns
        ]
        missing_prediction_columns = [
            col for col in prediction_columns if col not in outputs.columns
        ]
        extra_prediction_columns = [
            col for col in outputs.columns if col not in task.output_columns
        ]

        if missing_prediction_columns and len(missing_prediction_columns) == len(
            extra_prediction_columns
        ):
            outputs = outputs.rename(
                columns=dict(zip(extra_prediction_columns, missing_prediction_columns))
            )

    # Ensure we only keep required output columns from predictions
    common_cols = [col for col in task.output_columns if col in outputs.columns]
    outputs = outputs[common_cols]

    # Handle specific test ID column if providded and detected
    if (
        task.test_id_column is not None
        and task.test_id_column != NO_ID_COLUMN_IDENTIFIED
    ):
        test_ids = task.test_data[task.test_id_column]

        # Check if sample submission data is available for ID comparison
        if task.sample_submission_data is not None:
            output_ids = task.sample_submission_data[task.output_id_column]
            if not test_ids.equals(output_ids):
                print("WARNING: Test IDs and output IDs do not match!")

        # Ensure test ID column is included
        output_id_column = task.output_id_column or task.test_id_column
        if output_id_column not in outputs.columns:
            outputs[output_id_column] = test_ids.to_numpy()

    # Handle undetected ID columns
    missing_columns = [col for col in task.output_columns if col not in outputs.columns]
    if missing_columns:
        print(
            "WARNING: The following columns are not in predictions and will be treated as ID columns:"
            f"{missing_columns}"
        )

        for col in missing_columns:
            if sample_submission_matches and col in sample_submission.columns:
                outputs[col] = sample_submission[col].to_numpy()
                print(f"WARNING: Copied from sample submission for column '{col}'")
            elif task.test_data is not None:
                if col in task.test_data.columns:
                    # Copy from test data if available as a column
                    outputs[col] = task.test_data[col]
                    print(f"WARNING: Copied from test data for column '{col}'")
                elif (
                    col == task.test_data.index.name
                    or col in task.test_data.index.names
                ):
                    # Copy from test data index (e.g., timestamp column in time series)
                    outputs[col] = task.test_data.index
                    print(f"WARNING: Copied from test data index for column '{col}'")
                else:
                    # Generate unique integer values
                    outputs[col] = range(len(outputs))
                    print(
                        f"WARNING: Generated unique integer values for column '{col}'"
                        "as it was not found in test data"
                    )
            else:
                # Generate unique integer values
                outputs[col] = range(len(outputs))
                print(
                    f"WARNING: Generated unique integer values for column '{col}'"
                    "as it was not found in test data"
                )

    # Ensure columns are in the correct order
    outputs = outputs[task.output_columns]

    if sample_submission_matches:
        for column in task.output_columns:
            if column in sample_submission.columns:
                outputs[column] = _coerce_to_reference_dtype(
                    outputs[column], sample_submission[column]
                )

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
    output_filename: Annotated[
        Optional[str], typer.Option("--output-filename", help="Output CSV file path")
    ] = None,
) -> Tuple[PredictionTask, PredictionAssistant]:
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

        if output_filename is None:
            output_filename = (
                f"fedotllm-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        with open(output_filename, "w") as fp:
            make_prediction_outputs(task, predictions).to_csv(fp, index=False)

        print(f"Prediction complete! Outputs written to {output_filename}")

    if config.save_artifacts.enabled:
        artifacts_dir_name = f"{task.metadata['name']}_artifacts"
        if config.save_artifacts.append_timestamp:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir_name = (
                f"{task.metadata['name']}_artifacts_{current_timestamp}"
            )

        full_save_path = Path(config.save_artifacts.path) / artifacts_dir_name

        assistant.predictor.save_artifacts(str(full_save_path), task)

        print(
            f"Artifacts including transformed datasets and trained model saved at {full_save_path}",
            flush=True,
        )

    return task, assistant


def main():
    app = typer.Typer()
    app.command("run")(run_assistant)
    app()


if __name__ == "__main__":
    main()
