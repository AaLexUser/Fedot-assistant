import pandas as pd

from fedotllm.task import PredictionTask
from fedotllm.task_inference import task_inference as task_inference_module


def make_task() -> PredictionTask:
    return PredictionTask(
        filepaths=[],
        metadata={
            "description": "Competition dataset.",
            "label_column": "target",
        },
    )


def test_train_id_inference_uses_prompt_field_name_and_drops_column(monkeypatch):
    task = make_task()
    task.train_data = pd.DataFrame(
        {
            "series_id": [1, 2],
            "feature": [0.1, 0.2],
            "target": [10, 20],
        }
    )

    inference = task_inference_module.TrainIDColumnInference(llm=None)
    monkeypatch.setattr(
        inference,
        "_chat_and_parse_prompt_output",
        lambda: {"train_id_column": "series_id"},
    )
    monkeypatch.setattr(inference, "log_value", lambda *args, **kwargs: None)

    updated_task = inference.transform(task)

    assert updated_task.train_id_column == "series_id"
    assert "series_id" not in updated_task.train_data.columns
    assert updated_task.metadata["dropped_train_id_column"] is True


def test_test_id_inference_sets_none_when_test_data_missing():
    task = make_task()

    inference = task_inference_module.TestIDColumnInference(llm=None)

    updated_task = inference.transform(task)

    assert updated_task.test_id_column is None


def test_label_inference_falls_back_to_prompt_for_one_column_submission(monkeypatch):
    task = make_task()
    task.metadata["label_column"] = None
    task.train_data = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "target": [0, 1],
        }
    )
    task.sample_submission_data = pd.DataFrame({"target": [0, 1]})

    inference = task_inference_module.LabelColumnInference(llm=None)
    monkeypatch.setattr(
        inference,
        "_chat_and_parse_prompt_output",
        lambda: {"label_column": "target"},
    )
    monkeypatch.setattr(inference, "log_value", lambda *args, **kwargs: None)

    updated_task = inference.transform(task)

    assert updated_task.label_column == "target"
