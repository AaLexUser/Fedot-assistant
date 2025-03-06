"""A task encapsulates the data for a data science task or project. It contains descriptions, data, metadata."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from autogluon.tabular import TabularDataset
from .constants import OUTPUT, TEST, TRAIN


class TabularPredictionTask:
    """A task contains data and metadata for a tabular machine learning task, including datasets, metadata such as
    problem type, test_id_column, etc.
    """

    def __init__(
        self,
        filepaths: List[Path],
        metadata: Dict[str, Any],
        name: Optional[str] = "",
        description: Optional[str] = "",
        cache_data: bool = True,
    ):
        self.metadata: Dict[str, Any] = {
            "name": name,
            "description": description,
            "label_column": None,
            "problem_type": None,
            "eval_metric": None,  # string, keying Autogluon Tabular metrics
            "test_id_column": None,
        }

        self.metadata.update(metadata)

        self.filepaths = filepaths
        self.cache_data = cache_data

        # TODO: each data split can have multiple files
        self.dataset_mapping: Dict[str, Union[Path, pd.DataFrame, TabularDataset]] = {
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
        }

    def __repr__(self) -> str:
        return f"TabularPredictionTask(name={self.metadata['name']}, description={self.metadata['description'][:100]}, {len(self.dataset_mapping)} datasets)"

    @classmethod
    def from_path(
        cls, task_root_dir: Path, name: Optional[str] = None
    ) -> "TabularPredictionTask":
        # Get all filenames under task_root_dir
        task_data_filenames = []
        for root, _, files in os.walk(task_root_dir):
            for file in files:
                # Get the relative path
                relative_path = os.path.relpath(os.path.join(root, file), task_root_dir)
                task_data_filenames.append(relative_path)

        return cls(
            filepaths=[task_root_dir / fn for fn in task_data_filenames],
            metadata=dict(
                name=task_root_dir.name,
            ),
        )
