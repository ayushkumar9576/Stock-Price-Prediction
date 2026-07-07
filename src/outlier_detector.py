from __future__ import annotations
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaseOutlierDetector(ABC):
    @abstractmethod
    def detect_and_handle(self, df: pd.DataFrame)-> pd.DataFrame:
        raise NotImplementedError

class IQROutlierDetector(BaseOutlierDetector):
    def __init__(self, columns: list[str] | None = None, factor: float = 1.5, action: str = "clip"):
        action = action.lower()

        if action not in ("clip", "nan"):
            raise ValueError("Action Must be either 'clip' or 'nan'")
        if factor <= 0:
            raise ValueError("factor must be positive.")
        
        self._action = action
        self._factor = factor
        self._columns = columns
    
    def detect_and_handle(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Detecting and handling outlier in the dataset")
        df = df.copy()
        cols = self._columns if self._columns is not None else df.select_dtypes(include="number").columns.tolist()
        total = 0

        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self._factor * IQR
            upper = Q3 + self._factor * IQR

            mask = (df[col] < lower) | (df[col] > upper)
            n = int(mask.sum())
            total+=n

            if n>0:
                if self._action == "clip":
                    df[col] = df[col].clip(lower= lower, upper= upper)
                else:
                    df.loc[mask,col] = np.nan

                logger.debug(f"IQR [{col}]: {n} outlier(s) handled "
                    f"(action={self._action}, fence=[{lower:.4f}, {upper:.4f}])")        
        logger.info(f"IQROutlierDetector: handled {total} outlier(s) "
            f"across {cols} (factor={self._factor}, action={self._action}).")
    
        logger.info("Handled all the outlier in the dataset.")
        return df

class ZScoreOutlierDetector(BaseOutlierDetector):
    def __init__(self, columns: list[str] | None = None, threshold: float = 3.0, action: str = "clip") -> None:
        action = action.lower()

        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if action not in ("clip", "nan"):
            raise ValueError("action must be 'clip' or 'nan'.")
        self._columns = columns
        self._threshold = threshold
        self._action = action
    
    def detect_and_handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting and handling outlier in the dataset")
        df = df.copy()
        cols = self._columns or df.select_dtypes(include="number").columns.tolist()
        total = 0

        for col in cols:
            mean = df[col].mean()
            std = df[col].std()

            if not std or np.isnan(std):
                logger.debug(f"Skipping the column becasue Z-Score for the columns {col} is 0/NaN")
                continue

            z = (df[col] - mean) / std
            mask = z.abs() > self._threshold
            n = int(mask.sum())
            total += n

            if n > 0:
                lower = mean - self._threshold * std
                upper = mean + self._threshold * std
                if self._action == "clip":
                    df[col] = df[col].clip(lower=lower, upper=upper)
                else:
                    df.loc[mask, col] = np.nan

                logger.debug(f"Z-Score [{col}]: {n} outlier(s) handled "
                    f"(action={self._action}, |z| > {self._threshold})")

        logger.info(f"ZScoreOutlierDetector: handled {total} outlier(s) "
            f"across {cols} (threshold={self._threshold}, action={self._action}).")

        logger.info("Handled all the outlier in the dataset.")
        return df


class OutlierDetection:
    def __init__(self, strategy: BaseOutlierDetector) -> None:
        logger.info(f"Setting the strategy for Outlier Detection: {strategy.__class__.__name__}")
        self.strategy = strategy
    
    @property
    def strategy(self) -> BaseOutlierDetector:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseOutlierDetector) -> None:
        if not isinstance(strategy, BaseOutlierDetector):
            raise TypeError(f"Expected a BaseOutlierDetector, got {type(strategy)}")
        logger.info(f"OutlierDetector — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseOutlierDetector) -> None:
        self.strategy = strategy
    
    def detect_and_handle(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Handling the ")
        return self._strategy.detect_and_handle(df)