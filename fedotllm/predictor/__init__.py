from .autogluon import (
    AutogluonMultimodalPredictor,
    AutogluonTabularPredictor,
    AutogluonTimeSeriesPredictor,
)
from .fedot import (
    FedotMultiModalPredictor,
    FedotTabularPredictor,
    FedotTimeSeriesPredictor,
)
from .fedot_ind import (
    FedotIndustrialTabularPredictor,
    FedotIndustrialTimeSeriesPredictor,
)

__all__ = [
    "FedotTabularPredictor",
    "FedotMultiModalPredictor",
    "FedotTimeSeriesPredictor",
    "FedotIndustrialTabularPredictor",
    "FedotIndustrialTimeSeriesPredictor",
    "AutogluonTabularPredictor",
    "AutogluonMultimodalPredictor",
    "AutogluonTimeSeriesPredictor",
]
