from .base import Predictor
from typing import Any, Dict, Optional
from collections import defaultdict
from fedot.api.main import Fedot
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.data_detection import TextDataDetector
from fedot.core.data.data import InputData, array_to_input_data
from fedot.core.constants import DEFAULT_FORECAST_LENGTH
from fedot.core.repository.tasks import Task, TaskTypesEnum, TaskParams, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum
from ..task import PredictionTask
from ..utils import unpack_omega_config
from golem.core.dag.graph_utils import graph_structure
from fedotllm.tabular import TabularDataset
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from functools import partial

from ..constants import (
    ROC_AUC,
    LOG_LOSS,
    ACCURACY,
    F1,
    ROOT_MEAN_SQUARED_ERROR,
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    R2,
    BINARY,
    MULTICLASS,
    REGRESSION,
    CLASSIFICATION_PROBA_EVAL_METRIC
)

logger = logging.getLogger(__name__)

METRICS_TO_FEDOT = {
    ROC_AUC: "roc_auc",
    LOG_LOSS: "neg_log_loss",
    ACCURACY: "accuracy",
    F1: "f1",
    ROOT_MEAN_SQUARED_ERROR: "rmse",
    MEAN_SQUARED_ERROR: "mse",
    MEAN_ABSOLUTE_ERROR: "mae",
    R2: "r2"
}

PROBLEM_TO_FEDOT = {
    BINARY: "classification",
    MULTICLASS: "classification",
    REGRESSION: "regression"
}

def _get_task_with_params(problem: str, task_params: Optional[TaskParams] = None) -> Task:
        """ Creates Task from problem name and task_params"""
        if problem == 'ts_forecasting' and task_params is None:
            logger.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            task_params = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        try:
            return task_dict[problem]
        except ValueError:
            ValueError('Wrong type name of the given task')

#TODO: choose targer_size more reasonably
def _load_images_from_dataframe(df, image_column, target_size=(128, 128)):
        images = []
        for image_path in tqdm(df[image_column], desc="Loading images"):
            img = Image.open(image_path)
            img = img.resize(target_size)
            img_array = np.array(img)
            img_array = img_array.astype(np.float32) / sum(target_size) - 1  # Normalization
            
            # Ensure the image has 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale image
                img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to 3 channels
            elif img_array.shape[2] == 4:  # RGBA image
                img_array = img_array[:, :, :3]  # Drop the alpha channel
            images.append(img_array)
        
        images_array = np.array(images)
        return images_array
    
def prepare_multi_model_data(
    data: pd.DataFrame,
    idx_column: Optional[str],
    task: PredictionTask
) -> MultiModalData:
    # Verify critical columns exist
    if task.label_column not in data.columns:
        raise ValueError(f"Target column '{task.label_column}' not found in dataset")
    if task.images_column not in data.columns:
        raise ValueError(f"Images column '{task.images_column}' not found in dataset")

    target = data[task.label_column].to_numpy()
    idx = data[idx_column].to_numpy() if idx_column else np.arange(len(data))
    sources = {}
    
    # Prepare image data
    if task.images_column is None:
        logger.warning("Images column not found")
    else:
        data_img = InputData(
            idx=idx,
            task=_get_task_with_params(PROBLEM_TO_FEDOT[task.problem_type]),
            data_type=DataTypesEnum.image,
            features=_load_images_from_dataframe(data, task.images_column),
            target=target
        )
        data = data.drop(columns=[task.images_column, task.label_column])
        sources.update({"data_source_img": data_img})
    
    text_data_detector = TextDataDetector()
    text_columns = text_data_detector.define_text_columns(data)
    print(text_columns)
    if len(text_columns) > 0:
        data_text = text_data_detector.prepare_multimodal_data(data, text_columns)
        data_part_transformation_func = partial(array_to_input_data,
                                                    idx=idx, target_array=target, task=task)
        data = data.drop(columns=text_columns)
        
        text_sources = dict((text_data_detector.new_key_name(data_part_key),
                            data_part_transformation_func(features_array=data_part, data_type=DataTypesEnum.text))
                            for (data_part_key, data_part) in data_text.items()
                            if not text_data_detector.is_full_of_nans(data_part))
        sources.update(text_sources)
    
    #TODO: add sources to dict
    data_table = InputData(
        idx=idx,
        task=_get_task_with_params(PROBLEM_TO_FEDOT[task.problem_type]),
        data_type=DataTypesEnum.table,
        features=data.to_numpy(),
        target=target 
    )
    sources.update({"data_source_table": data_table})
    
    return MultiModalData(sources)

class FedotTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Fedot = None
        self.problem_type: str = None
    
    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotTabularPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type
        
        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric], 
            **unpack_omega_config(self.config.predictor_init_kwargs)  
        }
        
        predictor_fit_kwargs = self.config.predictor_fit_kwargs
        
        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")
        
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs
        }
        
        self.predictor = Fedot(**predictor_init_kwargs)
        self.predictor.fit(
            task.train_data, task.label_column, **unpack_omega_config(predictor_fit_kwargs)
        )
        
        self.metadata['graph_structure'] = graph_structure(self.predictor.current_pipeline)
        return self
    
    def predict(self, task: PredictionTask) -> TabularDataset:
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            predictions = self.predictor.predict_proba(
                task.test_data
            )
        else:
            predictions = self.predictor.predict(task.test_data)
        return pd.DataFrame(predictions, columns=[task.label_column])
        
    def save_artifacts(self, path: str):
        self.predictor.current_pipeline.save(path)
        
class FedotMultiModalPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.predictor: Fedot = None
        self.problem_type: str = None     
        
    def fit(self, task: PredictionTask, time_limit: Optional[float] = None) -> "FedotMultiModalPredictor":
        eval_metric = task.eval_metric
        self.problem_type = task.problem_type
                
        predictor_init_kwargs = {
            "problem": PROBLEM_TO_FEDOT[task.problem_type],
            "timeout": time_limit,
            "metric": METRICS_TO_FEDOT[eval_metric], 
            **unpack_omega_config(self.config.predictor_init_kwargs)  
        }
        
        train_data = prepare_multi_model_data(task.train_data, task.train_id_column, task)
        
        predictor_fit_kwargs = self.config.predictor_fit_kwargs
        
        logger.info("Fitting Fedot TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")
        
        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs
        }
        
        self.predictor = Fedot(**predictor_init_kwargs)
        self.predictor.fit(
            train_data, task.label_column, **unpack_omega_config(predictor_fit_kwargs)
        )
        
        self.metadata['graph_structure'] = graph_structure(self.predictor.current_pipeline)
        return self
    
    def predict(self, task: PredictionTask) -> TabularDataset:
        test_data = prepare_multi_model_data(task.test_data, task.test_id_column, task)
        
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.problem_type in [
            BINARY,
            MULTICLASS
        ]:
            predictions = self.predictor.predict_proba(
                test_data
            )
        else:
            predictions = self.predictor.predict(test_data)
        return pd.DataFrame(predictions, columns=[task.label_column])
        
    def save_artifacts(self, path: str):
        self.predictor.current_pipeline.save(path)