import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LSTM
from keras.losses import Huber
from keras.metrics import MeanAbsoluteError
from keras.models import Model as KerasModel, Sequential, load_model as Keras_load_model
from keras.optimizers import Adam

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    @property
    @abstractmethod
    def keras_model(self)-> KerasModel | None:
        pass

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, int]) -> None:
        if not isinstance(input_shape, tuple):
            raise TypeError("Input Shape must be a tuple")
        
        if len(input_shape)!=2:
            raise ValueError("Input shape must only contain 2 dimension: (Lookback, n_features)")
        
        for i in input_shape:
            if not isinstance(i, int) or isinstance(i, bool):
                raise TypeError("Input shape dimension must be Integer")
            if i <=0:
                raise ValueError("Input Shape dimension must be a positive number")
            
    @staticmethod
    def _validate_training_data(X: NDArray[np.number], y:NDArray[np.number], X_name: str = "X", y_name: str = "y")-> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{X_name} must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{y_name} must be a numpy array")

        if X.ndim != 3: 
            raise ValueError(f"{X_name} must be a 3d array with shape (n_sample, Lookback, n_features)")
        if y.ndim not in (1,2):
            raise ValueError(f"{y_name} must either be a 1d array or 2d single-columns array")        

        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"A 2D {y_name} array must contain exactly one target column")
        if y.shape[0] == 0:
            raise ValueError(f"{y_name} cannot be empty.")
        if X.shape[0] == 0:
            raise ValueError(f"{X_name} cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError(f"{X_name} must contain at least one time step.")
        if X.shape[2] == 0:
            raise ValueError(f"{X_name} must contain at least one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{X_name} and {y_name} must contain the same number of samples.\nReceived {X_name}={X.shape[0]} and {y_name}={y.shape[0]}.")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"{X_name} must contain numeric values.")
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"{y_name} must contain numeric values.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{X_name} must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{y_name} must contain only finite values.")
    
    @staticmethod
    def _validate_prediction_data(X: NDArray[np.number])-> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Predicted Data(X) must be a numpy array")

        if X.ndim != 3: 
            raise ValueError(f"Predicted Data(X) must be a 3d array with shape (n_sample, Lookback, n_features)")
        if X.shape[0] == 0:
            raise ValueError(f"Predicted Data(X) cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError(f"Predicted Data(X) must contain at least one time step.")
        if X.shape[2] == 0:
            raise ValueError(f"Predicted Data(X) must contain at least one feature.")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Predicted Data(X) must contain numeric values.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Predicted Data(X) must contain only finite values.")
        
    @abstractmethod
    def build(self, input_shape: tuple[int, int]) -> None:
        pass

    @abstractmethod
    def train(self, X: NDArray[np.number], y: NDArray[np.number], validation_data: tuple[NDArray[np.number], NDArray[np.number]] | None = None) -> Any:
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.number]) -> NDArray[np.number]:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        pass

