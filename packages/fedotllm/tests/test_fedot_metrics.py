from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedotllm.constants import REGRESSION, ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR
from fedotllm.predictor import fedot as fedot_module
from sklearn.metrics import mean_squared_log_error


class _FakePipeline:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.log = SimpleNamespace(log_or_raise=lambda *args, **kwargs: None)

    def predict(
        self,
        reference_data,
        output_mode="default",
        predictions_cache=None,
        fold_id=None,
    ):
        return OutputData(
            idx=reference_data.idx,
            features=reference_data.features,
            predict=self.predictions,
            task=reference_data.task,
            target=self.predictions,
            data_type=reference_data.data_type,
        )


def test_fedot_rmsle_metric_matches_sklearn():
    predictions = np.array([1.0, 2.0, 16.0])
    reference_data = InputData(
        idx=np.arange(3),
        features=np.zeros((3, 1)),
        target=np.array([1.0, 4.0, 9.0]),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
    )

    metric_value = fedot_module.FEDOT_RMSLE_METRIC(
        pipeline=_FakePipeline(predictions),
        reference_data=reference_data,
    )

    expected = float(
        np.sqrt(mean_squared_log_error(reference_data.target, predictions))
    )
    assert metric_value == pytest.approx(expected)


def test_fedot_tabular_predictor_passes_custom_rmsle_callable(monkeypatch):
    captured = {}

    class FakeFedot:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.current_pipeline = "fake-pipeline"

        def fit(self, features, target, **kwargs):
            captured["fit_features"] = features.copy()
            captured["fit_target"] = target.copy()
            captured["fit_kwargs"] = kwargs

    monkeypatch.setattr(fedot_module, "Fedot", FakeFedot)
    monkeypatch.setattr(fedot_module, "graph_structure", lambda pipeline: "fake-graph")

    predictor = fedot_module.FedotTabularPredictor(
        SimpleNamespace(predictor_init_kwargs={}, predictor_fit_kwargs={})
    )
    train_data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "target": [1.0, 4.0, 9.0],
        }
    )
    task = SimpleNamespace(
        eval_metric=ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
        problem_type=REGRESSION,
        train_data=train_data,
        label_column="target",
    )

    predictor.fit(task, time_limit=7)

    assert captured["init_kwargs"]["metric"] is fedot_module.FEDOT_RMSLE_METRIC
    assert captured["init_kwargs"]["problem"] == "regression"
    assert captured["init_kwargs"]["timeout"] == 7
    pd.testing.assert_frame_equal(
        captured["fit_features"],
        train_data.drop(columns=["target"]),
    )
    pd.testing.assert_series_equal(captured["fit_target"], train_data["target"])
    assert (
        predictor.metadata["predictor_init_kwargs"]["metric"]
        is fedot_module.FEDOT_RMSLE_METRIC
    )
