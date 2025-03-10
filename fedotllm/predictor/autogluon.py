"""Predictors solve tabular prediction tasks"""
import os
import joblib
import shutil
from collections import defaultdict
from typing import Any, Dict

from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from fedotllm.tabular import TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error

from ..task import PredictionTask
from ..utils import unpack_omega_config
from .base import Predictor
import logging
import numpy as np
from ..constants import (
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    CLASSIFICATION_PROBA_EVAL_METRIC,
    BINARY,
    MULTICLASS
)

logger = logging.getLogger(__name__)

def rmsle_func(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))

root_mean_square_logarithmic_error = make_scorer(
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    rmsle_func,
    optimum=0,
    greater_is_better=False
)

    
class AutogluonTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: TabularPredictor = None
        
    def save_dataset_details(self, task: PredictionTask) -> None:
        for key, data in (
            ("train", task.train_data),
            ("test", task.test_data)
        ):
            self.metadata["dataset_summery"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(data).to_dict()
            self.metadata["feature_missing_values"] = (data.isna().sum() / len(data)).to_dict()
        
    def fit(self, task, time_limit = None):
        """Trains an AutoGluon TabularPredictor with parsed arguments. Saves trained predictor
        to `self.predictor`
        
        Raises
        ------
        Exception
            TabularPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR:
            eval_metric = root_mean_square_logarithmic_error
            
        predictor_init_kwargs = {
            "learner_kwargs": {"ignored_columns": task.columns_in_train_but_not_test},
            "label": task.label_column,
            "problem_type": task.problem_type,
            "eval_metric": eval_metric,
            **unpack_omega_config(self.config.predictor_init_kwargs)
        }
        
        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)
        
        logger.info("Fitting AutoGluon TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")
        
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs
        }
        
        self.save_dataset_details(task)
        self.predictor = TabularPredictor(**predictor_init_kwargs).fit(
            task.train_data, **unpack_omega_config(predictor_fit_kwargs), time_limit=time_limit
        )
        
        self.metadata["leaderboard"] = self.predictor.leaderboard().to_dict()
        return self
    
    def predict(self, task: PredictionTask) -> TabularDataset:
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.predictor.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            return self.predictor.predict_proba(
                task.test_data, as_multiclass=(self.predictor.problem_type == MULTICLASS)
            )
        else:
            return self.predictor.predict(task.test_data)
        
    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data
        }
        
        ag_model_dir = self.predictor.path
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)
        
        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)
            
            local_model_dir = os.path.join(path, ag_model_dir)
            shutil.copytree(ag_model_dir, local_model_dir, dir_exist_ok=True)
            
class AutogluonMultimodalPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: MultiModalPredictor = None
        
    def save_dataset_details(self, task: PredictionTask) -> None:
        for key, data in (
            ("train", task.train_data),
            ("test", task.test_data)
        ):
            self.metadata["dataset_summery"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(data).to_dict()
            self.metadata["feature_missing_values"] = (data.isna().sum() / len(data)).to_dict()
        
    def fit(self, task, time_limit = None):
        """Trains an AutoGluon TabularPredictor with parsed arguments. Saves trained predictor
        to `self.predictor`
        
        Raises
        ------
        Exception
            TabularPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR:
            eval_metric = root_mean_square_logarithmic_error
            
        predictor_init_kwargs = {
            "label": task.label_column,
            "problem_type": task.problem_type,
            "eval_metric": eval_metric,
            **unpack_omega_config(self.config.predictor_init_kwargs)
        }
        
        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)
        
        logger.info("Fitting AutoGluon TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")
        
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs
        }
        
        self.save_dataset_details(task)
        self.predictor = MultiModalPredictor(**predictor_init_kwargs).fit(
            task.train_data, **unpack_omega_config(predictor_fit_kwargs), time_limit=time_limit
        )
        
        return self
    
    def predict(self, task: PredictionTask) -> TabularDataset:
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.predictor.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            return self.predictor.predict_proba(
                task.test_data, as_multiclass=(self.predictor.problem_type == MULTICLASS)
            )
        else:
            return self.predictor.predict(task.test_data)
        
    def save_artifacts(self, path: str, task: PredictionTask):
        artifacts = {
            "trained_model": self,
            "train_data": task.train_data,
            "test_data": task.test_data,
            "out_data": task.sample_submission_data
        }
        
        ag_model_dir = self.predictor.path
        full_save_path_pkl_file = f"{path}/artifacts.pkl"
        os.makedirs(path, exist_ok=True)
        
        with open(full_save_path_pkl_file, "wb") as f:
            joblib.dump(artifacts, f)
            
            local_model_dir = os.path.join(path, ag_model_dir)
            shutil.copytree(ag_model_dir, local_model_dir, dir_exist_ok=True)