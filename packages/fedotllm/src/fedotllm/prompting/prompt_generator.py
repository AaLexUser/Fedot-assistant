from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd

from ..constants import (
    DATA_EXTENSIONS,
    METRICS_DESCRIPTION,
    MULTICLASS,
    MULTIMODAL,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    NO_TIMESTAMP_COLUMN_IDENTIFIED,
    REGRESSION,
    TABULAR,
    TASK_TYPES,
    TIME_SERIES,
)
from ..utils import is_text_file
from .utils import get_outer_columns, parse_and_check_json


class PromptGenerator(ABC):
    fields = []

    def __init__(self, data_description: str = ""):
        self.data_description = data_description
        self.parser = self.create_parser()

    @property
    def system_prompt(self):
        return (
            "You are an expert assistant that parses information about data science tasks, "
            "such as data science competitions."
        )

    @property
    def basic_intro_prompt(self):
        return "The following section contain descriptive information about a data science task:"

    @property
    def data_description_prompt(self):
        return f"# Data Description\n{self.data_description}"

    @abstractmethod
    def generate_prompt(self) -> str:
        pass

    def get_field_parsing_prompt(self) -> str:
        return (
            f"Based on the above information, provide the correct values for the following fields strictly "
            f"in valid JSON format: {', '.join(self.fields)}.\n\n"
            "Important:\n"
            "1. Return only valid JSON. No extra explanations, text, or comments.\n"
            "2. Ensure that the output can be parsed by a JSON parser directly.\n"
            "3. Do not include any non-JSON text or formatting outside the JSON object.\n"
            '4. An example is {"<provided_field>": "<correct_value_for_the_field>"}'
        )

    def generate_chat_prompt(self):
        chat_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.generate_prompt()},
        ]

        return chat_prompt

    def create_parser(self):
        return partial(parse_and_check_json, expected_keys=self.fields)

    def read_file_safely(self, filename: Path) -> Union[str, None]:
        try:
            return filename.read_text()
        except UnicodeDecodeError:
            return None

    def build_file_preview_prompt(
        self, filenames: list[Path], header: str, allowed_suffixes: tuple[str, ...] = ()
    ) -> str:
        file_content_prompts = f"{header}\n\n"
        for filename in filenames:
            if is_text_file(filename) or filename.suffix in allowed_suffixes:
                content = self.read_file_safely(filename)
                if content is None:
                    continue

                truncated_contents = content[:100].strip()
                if len(content) > 100:
                    truncated_contents += "..."
                file_content_prompts += f"File:\n\n{filename.name} Truncated Content:\n{truncated_contents}\n\n"
        return file_content_prompts


class TaskTypePromptGenerator(PromptGenerator):
    fields = ["reasoning", "task_type"]

    def __init__(
        self, data_description: str, train_data: pd.DataFrame, label_column: str
    ):
        super().__init__(data_description)
        self.train_data = train_data
        self.label_column = label_column

    @property
    def train_sample_string(self) -> str:
        train_sample = self.train_data.head(10).to_markdown(index=False)
        return f"Train sample:\n\n{train_sample}"

    @property
    def label_column_string(self) -> str:
        if self.label_column is not None:
            return f"Label column: {self.label_column}"
        return "No label column found."

    def generate_prompt(self) -> str:
        task_type_instructions = f"""DETERMINE THE TASK TYPE

Analyze the data sample and description to identify the correct task_type from these options: {", ".join(TASK_TYPES)}

DECISION RULES (apply in this exact priority order):

1. TIME SERIES CHECK
   Return "{TIME_SERIES}" if ALL the following conditions are met:
   - There is a column containing dates, timestamps, or time values
   - The task involves forecasting or predicting future values
   - The data describes measurements taken over time
   - Look at the train sample: if rows represent sequential time periods, this is time_series

2. MULTIMODAL CHECK
   Return "{MULTIMODAL}" if you see:
   - Columns containing long text passages (sentences, paragraphs)
   - Columns with file paths to images (e.g., "/path/to/image.jpg")
   - Columns with mixed data types like text + structured data combined

3. DEFAULT
   Only return "{TABULAR}" if NEITHER of the above conditions apply.
"""

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                self.label_column_string,
                self.train_sample_string,
                task_type_instructions,
                self.get_field_parsing_prompt(),
            ]
        )


class ProblemTypePromptGenerator(PromptGenerator):
    fields = ["reasoning", "problem_type"]

    def __init__(
        self, data_description: str, train_data: pd.DataFrame, label_column: str
    ):
        super().__init__(data_description)
        self.train_data = train_data
        self.label_column = label_column

    @property
    def label_train_sample(self) -> str:
        col = self.train_data[self.label_column]
        label_sample = col.sample(n=min(10, len(col))).to_markdown(index=False)
        return f"Label column sample:\n\n{label_sample}"

    def generate_prompt(self) -> str:
        problem_types = [MULTICLASS, REGRESSION]
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                self.label_train_sample,
                (
                    "Based on the information provided, identify the correct problem_type to be used "
                    f"from among these KEYS: {', '.join(problem_types)}\n\n"
                    f"Response with the value {MULTICLASS} if the label is categorical, or {REGRESSION} if the label is continuous."
                ),
                self.get_field_parsing_prompt(),
            ]
        )


class DescriptionFileNamePromptGenerator(PromptGenerator):
    fields = ["data_description_file"]

    def __init__(self, filenames: list):
        super().__init__()
        self.filenames = filenames

    def generate_prompt(self) -> str:
        file_content_prompts = self.build_file_preview_prompt(
            filenames=list(map(Path, self.filenames)),
            header="# Available Files And Content in The File",
        )
        file_content_prompts += (
            "Please return the full path of the file that describes the problem settings, "
            f"and response with the value {NO_FILE_IDENTIFIED} if there's no such file."
        )
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
        )


class DataFilePromptGenerator(PromptGenerator):
    """Base for prompt generators that identify a single data file from a list of candidates."""

    question: str = ""

    def __init__(self, data_description: str, filenames: list[Path]):
        super().__init__(data_description)
        self.filenames = filenames

    def generate_prompt(self) -> str:
        file_list = self.build_file_preview_prompt(
            filenames=self.filenames,
            header="# Available Data Files",
            allowed_suffixes=tuple(DATA_EXTENSIONS),
        )
        file_list += (
            f"\n{self.question} "
            "Please return the full path of the data file as provided, "
            f"and respond with the value {NO_FILE_IDENTIFIED} if there's no such file."
        )
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                file_list,
                self.get_field_parsing_prompt(),
            ]
        )


class TrainDataFileNamePromptGenerator(DataFilePromptGenerator):
    fields = ["train_data"]
    question = (
        "Based on the data description and file previews, which file contains the training data? "
        "The training file is usually the largest and contains both feature columns and the target/label column. "
        "Look for filenames containing 'train' or files whose column headers include a label/target."
    )


class TestDataFileNamePromptGenerator(DataFilePromptGenerator):
    fields = ["test_data"]
    question = (
        "Based on the data description and file previews, which file contains the test data? "
        "The test file has similar column names to the training file but lacks the target/label column. "
        "Look for filenames containing 'test' or files with fewer columns than the training file."
    )


class SampleSubmissionDataFileNamePromptGenerator(DataFilePromptGenerator):
    fields = ["sample_submission_data"]
    question = (
        "Based on the data description and file previews, which file is the sample submission (output) file? "
        "The submission file typically has very few columns — usually just an ID column and the target column. "
        "Look for filenames containing 'submission', 'sample', 'output', or 'benchmark'."
    )


class StaticFeaturesFileNamePromptGenerator(DataFilePromptGenerator):
    fields = ["static_features_data"]
    question = (
        "Based on the data description and file previews, which file contains the static features data? "
        "Static features are the time-independent attributes (metadata) of a time series. "
        "These may include information such as: location where the time series was recorded, "
        "fixed properties of a product (brand, color, size), store ID or product ID. "
        "The file contains a table with features. "
        "Look for filenames containing 'static', 'metadata', or 'features'."
    )


class LabelColumnPromptGenerator(PromptGenerator):
    fields = ["label_column"]

    def __init__(self, data_description: str, column_names: list):
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the data description, which one of these columns is likely to be the label column:"
                    f"\n{', '.join(self.column_names)}"
                ),
                self.get_field_parsing_prompt(),
            ]
        )


class TimestampColumnPromptGenerator(PromptGenerator):
    fields = ["timestamp_column"]

    def __init__(self, data_description: str, column_names: list):
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the data description, which one of these columns is likely to be the timestamp column:"
                    f"\n{', '.join(self.column_names)}"
                    f"If no reasonable timestamp column is preset, response with the value {NO_TIMESTAMP_COLUMN_IDENTIFIED}"
                ),
                self.get_field_parsing_prompt(),
            ]
        )


class ForecastLengthPromptGenerator(PromptGenerator):
    fields = ["reasoning", "forecast_horizon"]

    def __init__(self, data_description: str):
        super().__init__(data_description)

    def generate_prompt(self) -> str:
        instructions = """INSTRUCTIONS:
Read the Data Description above and extract the forecast horizon (number of time steps to predict).

STEP 1: Find the phrase about predicting/forecasting ahead. Examples:
- "predict for NUMBER days" 
- "forecast NUMBER weeks ahead"
- "next NUMBER hours"
- "на NUMBER дня вперед" (Russian: for NUMBER days ahead)

STEP 2: Extract ONLY the NUMBER before the time unit.

STEP 3: Return that number as forecast_horizon."""

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                instructions,
                self.get_field_parsing_prompt(),
            ]
        )


class IDColumnPromptGenerator(PromptGenerator):
    fields = ["id_column"]

    def __init__(self, data_description: str, column_names: list, label_column: str):
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)
        self.label_column = label_column

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the data description, which one of these columns is likely to be the Id column:\n{', '.join(self.column_names)}",
                f"If no reasonable Id column is preset, for example if all the columns appear to be similarly named feature columns, "
                f"response with the value {NO_ID_COLUMN_IDENTIFIED}",
                f"ID columns can't be {self.label_column}",
                self.get_field_parsing_prompt(),
            ]
        )


class TestIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["test_id_column"]


class TrainIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["train_id_column"]


class OutputIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["output_id_column"]


class EvalMetricPromptGenerator(PromptGenerator):
    fields = ["eval_metric"]

    def __init__(self, data_description: str, metrics: str):
        super().__init__(data_description)
        self.metrics = metrics

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the information provided, identify the correct evaluation metric to be used from among these KEYS:\n"
                    f"{', '.join(self.metrics)}\n"
                    "The descriptions of these metrics are:\n"
                    f"{', '.join([METRICS_DESCRIPTION[metric] for metric in self.metrics])}\n"
                    "respectively."
                    "If the exact metric is not in the list provided, "
                    "then choose the metric that you think best approximates the one in the task description."
                    "Only respond with the exact names of the metrics mentioned in KEYS."
                    "Do not respond with the metric descriptions."
                ),
                self.get_field_parsing_prompt(),
            ]
        )
