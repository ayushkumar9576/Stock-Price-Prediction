import logging
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)

class BaseMissingValueStrategy(ABC):
    @abstractmethod
    def handle_missing_value(self, df: pd.DataFrame)-> pd.DataFrame:
        pass

class TimeSeriesImputer(BaseMissingValueStrategy):
    def __init__(self, columns: list[str] | None = None)-> None:
        self._columns = columns
    
    def handle_missing_value(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Handling the Missing Value using TimeSeriesImputer")
        df = df.copy()

        cols = self._columns if self._columns is not None else df.select_dtypes(include=["number"]).columns.to_list()
        
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {sorted(missing)}")

        before = df[cols].isna().sum().sum()

        if before==0:
            logger.info("TimeSeriesImputer: No missing values found")
            return df

        df[cols] = df[cols].ffill().bfill()

        after = df[cols].isna().sum().sum()

        logger.info(f"Before filling: {before} missing columns."
                    f" After filling: {after} missing columns")
        return df


class MeanImputer(BaseMissingValueStrategy):
    def __init__(self, columns: list[str] | None = None)-> None:
        self._columns = columns
    
    def handle_missing_value(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Handling the missing value using MeanImputer")
        df = df.copy()

        cols = self._columns if self._columns is not None else df.select_dtypes(include="number").columns.to_list() 

        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {sorted(missing)}")

        before = df[cols].isna().sum().sum()

        if before == 0:
            logger.info("No missing values found in the DataSet")
            return df
        
        df[cols] = df[cols].fillna(df[cols].mean())

        after = df[cols].isna().sum().sum()

        logger.info(f"Before filling: {before} missing columns."
                    f"After filling: {after} missing columns")

        return df


class DropMissingValues(BaseMissingValueStrategy):
    def __init__(self, axis: int = 0, thresh: int|None = None)-> None:
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (rows) or 1 (columns).")

        if thresh is not None and thresh < 0:
            raise ValueError("thresh must be greater than or equal to 0.")

        self.thresh = thresh
        self.axis = axis
    
    def handle_missing_value(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info(f"Dropping missing value with axis = {self.axis} and thresh = {self.thresh}")

        kwargs = {"axis": self.axis}
        if self.thresh is not None:
            kwargs["thresh"] = self.thresh

        cleaned = df.dropna(**kwargs)
        
        logger.info("Dropped Missing values")
        return cleaned

class MissingValueHandler:
    def __init__(self, strategy: BaseMissingValueStrategy)-> None:
        logger.info(f"Setting the strategy for Handling Missing Values: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def set_strategy(self, strategy: BaseMissingValueStrategy)-> None:
        logger.info(f"Changing the strategy for handling missing values: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def handle_missing_value(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Handling the missing values using the selected strategy")
        return self._strategy.handle_missing_value(df)