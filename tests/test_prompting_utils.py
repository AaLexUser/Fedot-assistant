from fedotllm.prompting.utils import _resolve_valid_value


def test_resolve_valid_value_preserves_parent_directory_prefix():
    valid_values = ["../data/train.csv"]

    resolved = _resolve_valid_value("../data/train.csv", valid_values)

    assert resolved == "../data/train.csv"


def test_resolve_valid_value_preserves_hidden_filename():
    valid_values = [".hidden_file.csv"]

    resolved = _resolve_valid_value(".hidden_file.csv", valid_values)

    assert resolved == ".hidden_file.csv"
