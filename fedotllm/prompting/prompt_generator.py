from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from ..utils import is_text_file
from ..constants import (
    NO_FILE_IDENTIFIED,
    PROBLEM_TYPES,
    NO_ID_COLUMN_IDENTIFIED,
    NO_TIMESTAMP_COLUMN_IDENTIFIED,
    METRICS_DESCRIPTION,
    TASK_TYPES,
    DATA_EXTENSIONS,
)
from .utils import get_outer_columns, parse_and_check_json
from functools import partial


class PromptGenerator(ABC):
    fields = []

    def __init__(self, data_description: str = ""):
        self.data_description = data_description
        self.parser = self.create_parser()

    @property
    def system_prompt(self):
        return (
            "You are an expert assistant that parses information about data science tasks,"
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
            "3. Do not include any non-JSON text or formatting outside the JSON object."
            '4. An example is \{"<provided_field>": "<correct_value_for_the_field>"\}'
        )

    def generate_chat_prompt(self):
        chat_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.generate_prompt()},
        ]

        return chat_prompt

    def create_parser(self):
        return partial(parse_and_check_json, expected_keys=self.fields)


class TaskTypePromptGenerator(PromptGenerator):
    fields = ["task_type"]

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the information provided, identify the correct task_type to be used "
                    f"from among these KEYS: {', '.join(TASK_TYPES)}"
                ),
                self.get_field_parsing_prompt(),
            ]
        )


class DescriptionFileNamePromptGenerator(PromptGenerator):
    fields = ["data_description_file", "evaluation_description_file"]

    def __init__(self, filenames: list):
        super().__init__()
        self.filenames = filenames

    def read_file_safely(self, filename: Path) -> Union[str, None]:
        try:
            return filename.read_text()
        except UnicodeDecodeError:
            return None

    def generate_prompt(self) -> str:
        file_content_prompts = "# Available Files And Content in The File\n\n"
        for filename in map(Path, self.filenames):
            if is_text_file(filename):
                content = self.read_file_safely(filename)
                if content is not None:
                    truncated_contents = content[:100].strip()
                    if len(content) > 100:
                        truncated_contents += "..."
                    file_content_prompts += f"File:\n\n{filename} Truncated Content:\n{truncated_contents}\n\n"
        file_content_prompts += (
            "Please return the full path of the file to describe the problem settings, "
            f"and response with the value {NO_FILE_IDENTIFIED} if there's no such file."
        )
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
        )


class DataFileNamePromptGenerator(PromptGenerator):
    fields = ["train_data", "test_data", "sample_submission_data"]

    def __init__(self, data_description: str, filenames: list):
        super().__init__(data_description)
        self.filenames = filenames

    def generate_prompt(self) -> str:
        file_content_prompts = "# Available Data File And Columns in The File\n\n"
        for filename in self.filenames:
            file_content_prompts += f"File:\n\n{filename}"

        file_content_prompts += (
            f"Based on the data description, what are the training, test, and output data? "
            "The output file may contain keywords such as benchmark, submission, or output. "
            "Please return the full path of the data files as provided, "
            f"and response with the value {NO_FILE_IDENTIFIED} if there's no such File."
        )

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
        )


class StaticFeaturesFileNamePromptGenerator(PromptGenerator):
    fields = ["static_features_data"]

    def __init__(
        self, data_description: str, filenames: list, data_description_file: str
    ):
        super().__init__(data_description)
        self.data_description_file = data_description_file
        self.filenames = filenames

    def read_file_safely(self, filename: Path) -> Union[str, None]:
        try:
            return filename.read_text()
        except UnicodeDecodeError:
            return None

    def generate_prompt(self) -> str:
        file_content_prompts = "# Available Data File And Content in The File\n\n"
        for filename in map(Path, self.filenames):
            if is_text_file(filename):
                content = self.read_file_safely(filename)
                if content is not None:
                    truncated_contents = content[:100].strip()
                    if len(content) > 100:
                        truncated_contents += "..."
                    file_content_prompts += f"File:\n\n{filename} Truncated Content:\n{truncated_contents}\n\n"

        file_content_prompts += (
            f"Based on the data description, what is the static features data? "
            "Static features are the time-independent attribures (metadata) of a time series. "
            "These may include information such as:\n"
            "- location, where the time series was recorded (country, state, city)\n"
            "- fixed properties of a product (brand name, color, size, weight)\n"
            "- store ID or product ID"
            "The file contains a table with features."
            "The static features file may contain keywords such as 'metadata', 'static_features'"
            f"File can't be {self.data_description_file}."
            f"File extention must be in {', '.join(DATA_EXTENSIONS)} "
            "Please return the full path of the data files as provided, "
            f"and response with the value {NO_FILE_IDENTIFIED} if there's no such File."
        )

        return "\n\n".join(
            [
                self.basic_intro_prompt,
                file_content_prompts,
                self.get_field_parsing_prompt(),
            ]
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


class ProblemTypePromptGenerator(PromptGenerator):
    fields = ["problem_type"]

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the information provided, identify the correct problem_type to be used "
                    f"from among these KEYS: {', '.join(PROBLEM_TYPES)}"
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
                    f"\n{', '.join(self.column_names)}",
                    f"If no reasonable timestamp column is preset, response with the value {NO_TIMESTAMP_COLUMN_IDENTIFIED}",
                ),
                self.get_field_parsing_prompt(),
            ]
        )


class ForecastLengthPromptGenerator(PromptGenerator):
    fields = ["forecast_horizon"]

    def __init__(self, data_description: str):
        super().__init__(data_description)

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                (
                    "Based on the data description, what is the forecast horizon (prediction_length) according to the task?"
                    "Please return an integer number, and response with the value {DEFAULT_FORECAST_HORIZON} if there's no information provided."
                ),
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
