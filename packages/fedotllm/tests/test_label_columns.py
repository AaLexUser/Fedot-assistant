from pathlib import Path

import pandas as pd
from fedotllm.task import PredictionTask


def test_prediction_task_infers_single_label_columns_from_sample_submission():
    task = PredictionTask(filepaths=[], metadata={"name": "single-target"})
    task.train_data = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "feature": [0.1, 0.2],
            "Survived": [0, 1],
        }
    )
    task.sample_submission_data = pd.DataFrame(
        {
            "PassengerId": [3, 4],
            "Survived": [0, 0],
        }
    )
    task.output_id_column = "PassengerId"

    assert task.label_columns == ["Survived"]


def test_prediction_task_infers_multilabel_columns_from_sample_submission():
    task = PredictionTask.from_path(Path("data/playground-series-s4e3").resolve())
    task.train_data = Path("data/playground-series-s4e3/train.csv").resolve()
    task.sample_submission_data = Path("data/playground-series-s4e3/sample_submission.csv").resolve()
    task.output_id_column = "id"

    assert task.label_columns == [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
