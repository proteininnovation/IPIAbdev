# models/__init__.py

from .xgboost import XGBoostModel
from .randomforest import RandomForestModel
from .cnn import CNNClassifier

available_models = {
    "xgboost": XGBoostModel,
    "rf": RandomForestModel,
    "cnn": CNNClassifier,
}