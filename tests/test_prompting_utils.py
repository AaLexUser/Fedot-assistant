import pytest

from fedotllm.exceptions import OutputParserException
from fedotllm.prompting.utils import _resolve_valid_value, parse_and_check_json


def test_resolve_valid_value_preserves_parent_directory_prefix():
    valid_values = ["../data/train.csv"]

    resolved = _resolve_valid_value("../data/train.csv", valid_values)

    assert resolved == "../data/train.csv"


def test_resolve_valid_value_preserves_hidden_filename():
    valid_values = [".hidden_file.csv"]

    resolved = _resolve_valid_value(".hidden_file.csv", valid_values)

    assert resolved == ".hidden_file.csv"


def test_parse_and_check_json_reports_full_payload_for_missing_key():
    with pytest.raises(OutputParserException) as exc_info:
        parse_and_check_json('{"wrong_key": "value"}', ["label_column"])

    assert str(exc_info.value) == (
        "Got invalid return object. Expected key `label_column` "
        "to be present, but got {'wrong_key': 'value'}"
    )
