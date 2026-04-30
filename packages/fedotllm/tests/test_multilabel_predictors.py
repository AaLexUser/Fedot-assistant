from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
from fedotllm.constants import MULTILABEL, ROC_AUC
from fedotllm.predictor import autogluon as autogluon_module
from fedotllm.predictor import fedot as fedot_module


def _build_multilabel_task():
    return SimpleNamespace(
        eval_metric=ROC_AUC,
        problem_type=MULTILABEL,
        label_columns=["label_a", "label_b"],
        train_data=pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0],
                "label_a": [0, 1, 0],
                "label_b": [1, 0, 1],
            }
        ),
        test_data=pd.DataFrame({"feature": [10.0, 20.0]}),
        sample_submission_data=None,
        columns_in_train_but_not_test=["label_a", "label_b"],
    )


def test_autogluon_tabular_predictor_trains_and_predicts_one_model_per_label(
    monkeypatch,
):
    captured = {"fit_calls": []}

    class FakeTabularPredictor:
        def __init__(self, **kwargs):
            self.label = kwargs["label"]
            self.problem_type = "binary"
            self.path = f"/tmp/{self.label}"

        def fit(self, train_data, **kwargs):
            captured["fit_calls"].append((self.label, train_data.copy(), kwargs))
            return self

        def predict_proba(self, test_data, as_multiclass=False):
            values = {"label_a": [0.1, 0.2], "label_b": [0.8, 0.7]}[self.label]
            return pd.Series(values, name=self.label, index=test_data.index)

        def leaderboard(self):
            return pd.DataFrame({"model": ["fake"]})

    monkeypatch.setattr(autogluon_module, "TabularPredictor", FakeTabularPredictor)
    predictor = autogluon_module.AutogluonTabularPredictor(
        SimpleNamespace(predictor_init_kwargs={}, predictor_fit_kwargs={})
    )
    task = _build_multilabel_task()

    predictor.fit(task, time_limit=12)
    predictions = predictor.predict(task)

    assert [label for label, _, _ in captured["fit_calls"]] == ["label_a", "label_b"]
    assert predictions.columns.tolist() == ["label_a", "label_b"]
    assert predictions["label_a"].tolist() == [0.1, 0.2]
    assert predictions["label_b"].tolist() == [0.8, 0.7]


def test_fedot_tabular_predictor_trains_and_predicts_one_model_per_label(monkeypatch):
    captured = {"fit_calls": []}

    class FakeFedot:
        def __init__(self, **kwargs):
            self.current_pipeline = "fake-pipeline"
            self.label = ""

        def fit(self, features, target, **kwargs):
            self.label = target.name
            captured["fit_calls"].append((self.label, features.copy(), target.copy()))

        def predict_proba(self, features):
            values = {"label_a": [0.3, 0.4], "label_b": [0.9, 0.6]}[cast(str, self.label)]
            return np.asarray(values).reshape(-1, 1)

    monkeypatch.setattr(fedot_module, "Fedot", FakeFedot)
    monkeypatch.setattr(fedot_module, "graph_structure", lambda pipeline: "fake-graph")

    predictor = fedot_module.FedotTabularPredictor(SimpleNamespace(predictor_init_kwargs={}, predictor_fit_kwargs={}))
    task = _build_multilabel_task()

    predictor.fit(task, time_limit=1)
    predictions = predictor.predict(task)

    assert [label for label, _, _ in captured["fit_calls"]] == ["label_a", "label_b"]
    assert predictions.columns.tolist() == ["label_a", "label_b"]
    assert predictions["label_a"].tolist() == [0.3, 0.4]
    assert predictions["label_b"].tolist() == [0.9, 0.6]
