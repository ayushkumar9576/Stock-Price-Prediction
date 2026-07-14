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
    def split(self, df: pd.DataFrame, feature_cols: list[str], target_column: str = "Close", ratio: float = 0.7)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame)-> Self:
        pass

    @abstractmethod
    def transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame | None = None)-> tuple[np.ndarray, np.ndarray] | np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, scaled: np.ndarray)-> np.ndarray:
        pass

class MinMaxDataScaler(BaseDataSplitterAndProcessingClass):
    def __init__(self, feature_range: tuple[float, float] = (0,1))-> None:
        self._feature_scaler = MinMaxScaler(feature_range=feature_range)
        self._target_scaler = MinMaxScaler(feature_range=feature_range)

        self._is_fitted = False

        self._feature_columns: list[str] | None = None
        self._target_column: str | None = None
    
    def split(self, df: pd.DataFrame, feature_cols: list[str],target_column: str = "Close", ratio: float = 0.7)-> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame, pd.DataFrame]:
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1.")
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise KeyError(f"Feature columns not found: {sorted(missing)}")
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
        
        logger.info(f"Spliting the DataSet using the ration: {ratio}")
        size = len(df)
        split_idx = int(size*ratio)

        X_train = df[feature_cols].iloc[:split_idx].copy()
        X_test = df[feature_cols].iloc[split_idx:].copy()

        y_train = df[[target_column]].iloc[:split_idx].copy()
        y_test = df[[target_column]].iloc[split_idx:].copy()

        logger.info(f"Split → train: {X_train.shape[0]} rows | test: {X_test.shape[0]} rows | ratio: {ratio:.0%}")

        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame)-> Self:
        if X_train.empty:
            raise ValueError("Training data cannot be empty.")
        if y_train.empty:
            raise ValueError("Testing data cannot be empty.")
        
        self._feature_scaler.fit(X_train)
        self._target_scaler.fit(y_train)

        self._feature_columns = list(X_train.columns)
        self._target_column = y_train.columns[0]

        self._is_fitted = True
        logger.debug("Feature and target scalers fitted successfully.")
        return self
    
    def transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame | None = None)-> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        
        if X_train.empty:
            raise ValueError("Input data cannot be empty.")

        if list(X_train.columns) != self._feature_columns:
            raise KeyError(f"Expected columns {self._feature_columns}, got {list(X_train.columns)}.")
        
        X_scaled = self._feature_scaler.transform(X_train)

        if y_train is None:
            logger.info("Feature data transformed Sucessfully")
            return X_scaled
        
        if y_train.empty:
            raise ValueError("Target data cannot be empty.")
        if list(y_train.columns) != [self._target_column]:
            raise KeyError(f"Expected columns {self._target_column}, got {list(y_train.columns)}.")
        
        y_scaled = self._target_scaler.transform(y_train)

        logger.info("Feature and target data transformed.")

        return X_scaled, y_scaled

    def inverse_transform(self, scaled: np.ndarray)-> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        
        scaled = np.asarray(scaled)

        if scaled.size == 0:
            raise ValueError("Input array cannot be empty.")

        if scaled.ndim == 1:
            scaled = scaled.reshape(-1, 1)

        logger.info("Converted the data back to original form")
        return self._target_scaler.inverse_transform(scaled)
    
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
    
    def split(self, df: pd.DataFrame, feature_cols: list[str], target_column: str = "Close", ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting the DataSet with the selected Strategy")
        return self._strategy.split(df, feature_cols= feature_cols, target_column=target_column, ratio=ratio)

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Self:
        logger.info("Fitting the Scaler on the training data only")
        self._strategy.fit(X_train=X_train, y_train=y_train)
        return self

    def transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame|None = None) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        logger.info("Scaling the training data with the fitted scaler")
        return self._strategy.transform(X_train, y_train)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        logger.info("Converting the scaled value back to the original value")
        return self._strategy.inverse_transform(scaled)