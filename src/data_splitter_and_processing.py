"""
STRATEGIES:
    1. TrainTestSplitter  — Splits the dataset into training and testing subsets.
    2. MinMaxDataScaler   — Scales numerical values to a specified range using Min-Max normalization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Self
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class BaseDataSplitterAndProcessingClass(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, columns: str = "Close", ratio: float = 0.7)-> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def fit(self, data_training: pd.DataFrame)-> Self:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame)-> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, scaled: np.ndarray)-> np.ndarray:
        pass

class MinMaxDataScaler(BaseDataSplitterAndProcessingClass):
    def __init__(self, feature_range: tuple[float, float] = (0,1))-> None:
        self._scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
        self._columns: list[str] | None = None
    
    def split(self, df: pd.DataFrame, columns: str = "Close", ratio: float = 0.7)-> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1.")
        if columns not in df.columns:
            raise KeyError(f"Column '{columns}' not found in DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        logger.info(f"Spliting the DataSet using the ration: {ratio}")
        size = len(df)
        split_idx = int(size*ratio)
        data_training = df[[columns]].iloc[:split_idx]
        data_testing = df[[columns]].iloc[split_idx:]

        logger.info(f"Split → train: {split_idx} rows | test: {size - split_idx} rows | ratio: {ratio:.0%}")

        return data_training, data_testing
    
    def fit(self, data_training: pd.DataFrame)-> Self:
        if data_training.empty:
            raise ValueError("Training data cannot be empty.")
        
        self._scaler.fit(data_training)
        self._columns = list(data_training.columns)
        self._is_fitted = True
        logger.debug("MinMaxScaler fitted on training data.")
        return self
    
    def transform(self, data: pd.DataFrame)-> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        
        if data.empty:
            raise ValueError("Input data cannot be empty.")

        if list(data.columns) != self._columns:
            raise KeyError(f"Expected columns {self._columns}, got {list(data.columns)}.")

        logger.info("Transformed the training data using the fitted scaler")
        return self._scaler.transform(data)
        
    def inverse_transform(self, scaled: np.ndarray)-> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        
        scaled = np.asarray(scaled)

        if scaled.size == 0:
            raise ValueError("Input array cannot be empty.")

        if scaled.ndim == 1:
            scaled = scaled.reshape(-1, 1)

        logger.info("Converted the data back to original form")
        return self._scaler.inverse_transform(scaled)
    
class DataPreprocessorAndSplitting:
    def __init__(self, strategy: BaseDataSplitterAndProcessingClass)-> None:
        if not isinstance(strategy, BaseDataSplitterAndProcessingClass):
            raise TypeError(f"Expected BaseDataSplitterAndProcessingClass, got {type(strategy)}")
        logger.info(f"Setting the strategy for Data Scaling: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    @property
    def strategy(self) -> BaseDataSplitterAndProcessingClass:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseDataSplitterAndProcessingClass) -> None:
        if not isinstance(strategy, BaseDataSplitterAndProcessingClass):
            raise TypeError(f"Expected a BaseDataSplitterAndProcessingClass, got {type(strategy)}")
        
        logger.info(f"Scaling Technique — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseDataSplitterAndProcessingClass) -> None:
        self.strategy = strategy
    
    def split(self, df: pd.DataFrame, column: str = "Close", ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting the DataSet with the selected Strategy")
        return self._strategy.split(df, columns=column, ratio=ratio)

    def fit(self, data_training: pd.DataFrame) -> Self:
        logger.info("Fitting the Scaler on the training data only")
        self._strategy.fit(data_training)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        logger.info("Scaling the training data with the fitted scaler")
        return self._strategy.transform(data)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        logger.info("Converting the scaled value back to the original value")
        return self._strategy.inverse_transform(scaled)