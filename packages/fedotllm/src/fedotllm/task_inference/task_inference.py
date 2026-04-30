import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union, cast

from ..constants import (
    BINARY,
    DATA_EXTENSIONS,
    DEFAULT_FORECAST_HORIZON,
    METRICS_BY_PROBLEM_TYPE,
    METRICS_DESCRIPTION,
    MULTILABEL,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    NO_TIMESTAMP_COLUMN_IDENTIFIED,
    OUTPUT,
    PROBLEM_TYPES,
    TASK_TYPES,
    TEST,
    TIME_SERIES,
    TRAIN,
)
from ..exceptions import OutputParserException
from ..llm import AssistantChatOpenAI
from ..prompting import (
    DescriptionFileNamePromptGenerator,
    EvalMetricPromptGenerator,
    ForecastLengthPromptGenerator,
    LabelColumnPromptGenerator,
    OutputIDColumnPromptGenerator,
    ProblemTypePromptGenerator,
    PromptGenerator,
    SampleSubmissionDataFileNamePromptGenerator,
    StaticFeaturesFileNamePromptGenerator,
    TaskTypePromptGenerator,
    TestDataFileNamePromptGenerator,
    TestIDColumnPromptGenerator,
    TimestampColumnPromptGenerator,
    TrainDataFileNamePromptGenerator,
    TrainIDColumnPromptGenerator,
)
from ..task import PredictionTask

logger = logging.getLogger(__name__)


def _get_tabular_filenames(paths: Iterable[Union[str, Path]]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).suffix.lower() in DATA_EXTENSIONS]


class TaskInference:
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    max_parse_retries: int = 3

    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm: AssistantChatOpenAI = llm
        self.fallback_value = None
        self.ignored_value: list[str] = []

    def initialize_task(self, task):
        self.prompt_generator: Optional[PromptGenerator] = None
        self.valid_values = None

    def log_value(self, key: str, value: Any, max_width: int = 1600) -> None:
        """Logs a key-value pair with formatted output"""
        if not value:
            logger.info(f"WARNING: The {key} of the task, it is set to None.")
            return

        prefix = key
        value_str = str(value).replace("\n", "\\n")
        if len(prefix) + len(value_str) > max_width:
            value_str = value_str[: max_width - len(prefix) - 3] + "..."

        bold_start = "\033[1m"
        bold_end = "\033[0m"

        logger.info(f"{bold_start}{prefix}{bold_end}: {value_str}")

    def transform(self, task: PredictionTask) -> PredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        for k, v in parser_output.items():
            if "reasoning" in k:
                continue
            if v in self.ignored_value:
                v = None
            self.log_value(k, v)
            setattr(task, k, self.post_process(task=task, value=v))
        return task

    def post_process(self, task, value):
        return value

    def _chat_and_parse_prompt_output(self) -> Dict[str, str]:
        assert self.prompt_generator is not None, "prompt_generator is not initialized"
        chat_prompt = self.prompt_generator.generate_chat_prompt()
        logger.debug(f"LLM chat_prompt:\n{chat_prompt}")

        last_error: OutputParserException | None = None
        retry_temperatures = [0.3, 0.5]
        original_temperature = self.llm.temperature

        for attempt in range(self.max_parse_retries):
            output = ""
            try:
                if attempt > 0:
                    temp = retry_temperatures[min(attempt - 1, len(retry_temperatures) - 1)]
                    self.llm.temperature = temp
                    logger.warning(
                        f"Retry {attempt}/{self.max_parse_retries - 1} "
                        f"(temperature={temp}) after OutputParserException: {last_error}"
                    )

                output = self.llm.invoke(chat_prompt)
                logger.debug(f"LLM output:\n{output}")
                parsed_output = self.prompt_generator.parser(
                    output,
                    valid_values=self.valid_values,
                    fallback_value=self.fallback_value,
                )
                return parsed_output
            except OutputParserException as e:
                last_error = e
                chat_prompt = chat_prompt + [
                    {"role": "assistant", "content": output},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed: {e}\n"
                            "Please fix the error and try again. "
                            "Return only valid JSON with the correct keys and values."
                        ),
                    },
                ]
            finally:
                self.llm.temperature = original_temperature

        logger.error(f"Failed to parse output after {self.max_parse_retries} attempts: {last_error}")
        logger.error(self.llm.describe())
        if last_error is None:
            raise RuntimeError("Failed to parse output without parser exception")
        raise last_error


class DescriptionFileNameInference(TaskInference):
    """Uses an LLM to locate the filenames of description files."""

    def initialize_task(self, task: PredictionTask):
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.prompt_generator = DescriptionFileNamePromptGenerator(filenames=filenames)

    def _read_descriptions(self, parser_output: Mapping[str, Union[str, Iterable[str]]]) -> str:
        description_parts = []
        for key, file_paths in parser_output.items():
            if isinstance(file_paths, str):
                file_paths = [file_paths]  # Convert single string to list

            for file_path in file_paths:
                if file_path == NO_FILE_IDENTIFIED:
                    continue
                else:
                    try:
                        with open(file_path, "r") as file:
                            content = file.read()
                            description_parts.append(f"{key}: {content}")
                    except FileNotFoundError:
                        continue
                    except IOError:
                        continue
        return "\n\n".join(description_parts)

    def transform(self, task: PredictionTask) -> PredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        descriptions_read = self._read_descriptions(parser_output)
        if descriptions_read:
            task.metadata["description"] = descriptions_read
        if "data_description_file" in parser_output.keys():
            task.data_description_file = parser_output["data_description_file"]
            self.log_value("data_description_file", parser_output["data_description_file"])
        self.log_value("description", descriptions_read)
        return task


class DataFileInference(TaskInference):
    """Base class for inferences that identify a single data file.

    Subclasses specify which files to exclude (already identified) and which
    prompt generator to use.
    """

    prompt_generator_class: type[PromptGenerator] | None = None

    def _get_excluded_paths(self, task: PredictionTask) -> set[Path]:
        """Return resolved paths of files already assigned to the task."""
        return set()

    def initialize_task(self, task: PredictionTask):
        excluded = self._get_excluded_paths(task)
        filenames = [p for p in _get_tabular_filenames(task.filepaths) if p.resolve() not in excluded]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.ignored_value = [NO_FILE_IDENTIFIED]
        self._remaining_filenames = filenames
        prompt_generator_class = self.prompt_generator_class
        assert prompt_generator_class is not None
        self.prompt_generator = cast(
            PromptGenerator,
            cast(Any, prompt_generator_class)(data_description=task.metadata["description"], filenames=filenames),
        )

    def transform(self, task: PredictionTask) -> PredictionTask:
        self.initialize_task(task)
        if not self._remaining_filenames:
            assert self.prompt_generator is not None
            field = cast(Any, self.prompt_generator).fields[0]
            self.log_value(field, None)
            setattr(task, field, None)
            return task
        return super().transform(task)


class TrainDataFileNameInference(DataFileInference):
    """Identifies the training data file."""

    prompt_generator_class = TrainDataFileNamePromptGenerator


class TestDataFileNameInference(DataFileInference):
    """Identifies the test data file, excluding the already-identified train file."""

    prompt_generator_class = TestDataFileNamePromptGenerator

    def _get_excluded_paths(self, task: PredictionTask) -> set[Path]:
        excluded = set()
        train_path = task.files_mapping.get(TRAIN)
        if train_path is not None:
            excluded.add(train_path.resolve())
        return excluded


class SampleSubmissionDataFileNameInference(DataFileInference):
    """Identifies the sample submission file, excluding train and test files."""

    prompt_generator_class = SampleSubmissionDataFileNamePromptGenerator

    def _get_excluded_paths(self, task: PredictionTask) -> set[Path]:
        excluded = set()
        for key in (TRAIN, TEST):
            path = task.files_mapping.get(key)
            if path is not None:
                excluded.add(path.resolve())
        return excluded


class StaticFeaturesFileNameInference(DataFileInference):
    """Identifies the static features file, excluding train, test, and output files."""

    prompt_generator_class = StaticFeaturesFileNamePromptGenerator

    def _get_excluded_paths(self, task: PredictionTask) -> set[Path]:
        """Exclude train, test, and output files that are already identified."""
        excluded = set()
        for key in (TRAIN, TEST, OUTPUT):
            path = task.files_mapping.get(key)
            if path is not None:
                excluded.add(path.resolve())
        return excluded


class TaskTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = TASK_TYPES
        self.prompt_generator = TaskTypePromptGenerator(
            data_description=task.metadata["description"],
            train_data=task.train_data,
            label_column=task.label_column,
        )


class ProblemTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = PROBLEM_TYPES
        self.prompt_generator = ProblemTypePromptGenerator(
            data_description=task.metadata["description"],
            train_data=task.train_data,
            label_column=task.label_column,
        )

    def transform(self, task: PredictionTask) -> PredictionTask:
        self.initialize_task(task)
        if task.task_type == TIME_SERIES:
            task.problem_type = TIME_SERIES
            return task
        if len(task.label_columns) > 1:
            task.problem_type = MULTILABEL
            return task
        if task.label_column is not None and task.train_data is not None:
            unique_values = task.train_data[task.label_column].unique()
            if len(unique_values) == 2:
                task.problem_type = BINARY
                return task
        return super().transform(task)


class LabelColumnInference(TaskInference):
    def transform(self, task: PredictionTask) -> PredictionTask:
        if task.sample_submission_data is not None:
            label_columns = task.label_columns
            self.log_value("label_columns", label_columns)
            task.label_columns = label_columns
            return task
        return super().transform(task)

    def initialize_task(self, task):
        column_names = list(task.train_data.columns)
        # Exclude ID columns from being considered as label columns
        id_columns = [col for col in [task.train_id_column, task.test_id_column] if col is not None]
        valid_columns = [col for col in column_names if col not in id_columns]
        self.valid_values = valid_columns
        self.prompt_generator = LabelColumnPromptGenerator(
            data_description=task.metadata["description"], column_names=valid_columns
        )

    def post_process(self, task, value):
        return [value]


class TimestampColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.train_data.columns)
        self.valid_values = column_names + [NO_TIMESTAMP_COLUMN_IDENTIFIED]
        self.prompt_generator = TimestampColumnPromptGenerator(
            data_description=task.metadata["description"], column_names=column_names
        )


class ForecastHorizonInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = None
        self.fallback_value = DEFAULT_FORECAST_HORIZON
        self.prompt_generator = ForecastLengthPromptGenerator(data_description=task.metadata["description"])

    def post_process(self, task, value):
        """Validate and convert forecast horizon to a positive integer."""
        horizon = None
        if isinstance(value, int):
            horizon = value
        elif isinstance(value, str):
            try:
                horizon = int(value)
            except ValueError:
                logger.warning(f"Could not convert forecast_horizon '{value}' to integer")
                return self.fallback_value

        if horizon is None or horizon <= 0:
            logger.warning(f"Invalid forecast_horizon value: {value}. Using default: {self.fallback_value}")
            return self.fallback_value

        return horizon


class BaseIDColumnInference(TaskInference):
    data_key: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_values = []
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_generator = None

    def get_data(self, task):
        raise NotImplementedError()

    def get_prompt_generator(self) -> type[PromptGenerator]:
        raise NotImplementedError()

    def process_id_column(self, task, id_column):
        raise NotImplementedError()

    def has_data_source(self, task: PredictionTask) -> bool:
        return self.data_key is not None and task.dataset_mapping[self.data_key] is not None

    def initialize_task(self, task, description=None):
        data = self.get_data(task)
        column_names = list(data.columns)
        # Assume ID column can only appear in first 3 columns
        if len(column_names) >= 3:
            column_names = column_names[:3]
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        if not description:
            description = task.metadata["description"]
        prompt_generator_factory = self.get_prompt_generator()
        self.prompt_generator = cast(
            PromptGenerator,
            cast(Any, prompt_generator_factory)(
                data_description=description,
                column_names=column_names,
                label_column=task.label_column,
            ),
        )

    def transform(self, task: PredictionTask) -> PredictionTask:
        id_column_name = cast(Any, self.get_prompt_generator()).fields[0]
        if not self.has_data_source(task):
            setattr(task, id_column_name, None)
            return task

        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()

        if parser_output[id_column_name] == NO_ID_COLUMN_IDENTIFIED:
            logger.warning(
                "Failed to infer ID column with data descriptions. Retry the inference without data descriptions."
            )
            self.initialize_task(
                task,
                description="Missing data description. Please infer the ID column based on given column names.",
            )
            parser_output = self._chat_and_parse_prompt_output()

        id_column = parser_output[id_column_name]
        id_column = self.process_id_column(task, id_column)
        self.log_value(id_column_name, id_column)
        setattr(task, id_column_name, id_column)
        return task


class TestIDColumnInference(BaseIDColumnInference):
    data_key = TEST

    def get_data(self, task):
        return task.test_data

    def get_prompt_generator(self):
        return TestIDColumnPromptGenerator

    def process_id_column(self, task, id_column):
        if task.output_id_column != NO_ID_COLUMN_IDENTIFIED:
            # if output data has id column but test data does not
            if id_column == NO_ID_COLUMN_IDENTIFIED:
                if task.output_id_column not in task.test_data:
                    id_column = task.output_id_column
                else:
                    id_column = "id_column"
                if task.sample_submission_data is not None:
                    new_test_data = task.test_data.copy()
                    new_test_data[id_column] = task.sample_submission_data[task.output_id_column]
                    task.test_data = new_test_data
        return id_column


class TrainIDColumnInference(BaseIDColumnInference):
    data_key = TRAIN

    def get_data(self, task):
        return task.train_data

    def get_prompt_generator(self):
        return TrainIDColumnPromptGenerator

    def process_id_column(self, task, id_column):
        return id_column


class OutputIDColumnInference(BaseIDColumnInference):
    data_key = OUTPUT

    def get_data(self, task):
        return task.sample_submission_data

    def get_prompt_generator(self):
        return OutputIDColumnPromptGenerator

    def process_id_column(self, task, id_column):
        return id_column


class DropIDColumnInference(TaskInference):
    def transform(self, task: PredictionTask) -> PredictionTask:
        id_column = task.train_id_column
        if (
            id_column in (None, NO_ID_COLUMN_IDENTIFIED)
            or task.dataset_mapping[TRAIN] is None
            or id_column not in task.train_data.columns
        ):
            return task

        task.train_data = task.train_data.drop(columns=[id_column])
        logger.info(f"Dropping ID column {id_column} from training data.")
        task.metadata["dropped_train_id_column"] = True
        return task


class EvalMetricInference(TaskInference):
    def initialize_task(self, task):
        problem_type = task.problem_type
        self.metrics = (
            list(METRICS_DESCRIPTION.keys()) if problem_type is None else list(METRICS_BY_PROBLEM_TYPE[problem_type])
        )
        self.valid_values = self.metrics
        if problem_type:
            self.fallback_value = METRICS_BY_PROBLEM_TYPE[problem_type][0]
        self.prompt_generator = EvalMetricPromptGenerator(
            data_description=task.metadata["description"], metrics=self.metrics
        )
