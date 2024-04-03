import pandas as pd
import numpy as np
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelEvaluationconfig:
    pass
class ModelEvaluation:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)
