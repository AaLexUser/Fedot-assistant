import pandas as pd
from fedotllm import make_prediction_outputs
from fedotllm.task import PredictionTask


def _build_task(sample_submission_data: pd.DataFrame) -> PredictionTask:
    task = PredictionTask(filepaths=[], metadata={"name": "test-task"})
    task.test_data = pd.DataFrame(
        {
            "PassengerId": [892, 893],
            "feature": [1.0, 2.0],
        }
    )
    task.sample_submission_data = sample_submission_data
    task.test_id_column = "PassengerId"
    task.output_id_column = "PassengerId"
    task.label_column = "Survived"
    return task


def test_make_prediction_outputs_uses_sample_submission_schema_for_single_target():
    task = _build_task(
        pd.DataFrame(
            {
                "PassengerId": pd.Series([892, 893], dtype="int64"),
                "Survived": pd.Series([0, 0], dtype="int64"),
            }
        )
    )

    predictions = pd.DataFrame({"prediction": [1.0, 0.0]})

    outputs = make_prediction_outputs(task, predictions)

    assert outputs.columns.tolist() == ["PassengerId", "Survived"]
    assert outputs["PassengerId"].tolist() == [892, 893]
    assert outputs["Survived"].tolist() == [1, 0]
    assert pd.api.types.is_integer_dtype(outputs["Survived"])


def test_make_prediction_outputs_remaps_multiple_prediction_columns_by_position():
    task = _build_task(
        pd.DataFrame(
            {
                "PassengerId": pd.Series([892, 893], dtype="int64"),
                "No": pd.Series([0.0, 0.0], dtype="float64"),
                "Yes": pd.Series([0.0, 0.0], dtype="float64"),
            }
        )
    )

    predictions = pd.DataFrame(
        {
            "class_0": [0.8, 0.3],
            "class_1": [0.2, 0.7],
        }
    )

    outputs = make_prediction_outputs(task, predictions)

    assert outputs.columns.tolist() == ["PassengerId", "No", "Yes"]
    assert outputs["PassengerId"].tolist() == [892, 893]
    assert outputs["No"].tolist() == [0.8, 0.3]
    assert outputs["Yes"].tolist() == [0.2, 0.7]


def test_make_prediction_outputs_keeps_values_when_dtype_coercion_fails():
    task = _build_task(
        pd.DataFrame(
            {
                "PassengerId": pd.Series([892, 893], dtype="int64"),
                "Survived": pd.Series([0, 0], dtype="int64"),
            }
        )
    )

    predictions = pd.DataFrame({"prediction": [0.5, 1.5]})

    outputs = make_prediction_outputs(task, predictions)

    assert outputs.columns.tolist() == ["PassengerId", "Survived"]
    assert outputs["PassengerId"].tolist() == [892, 893]
    assert outputs["Survived"].tolist() == [0.5, 1.5]
    assert pd.api.types.is_float_dtype(outputs["Survived"])
