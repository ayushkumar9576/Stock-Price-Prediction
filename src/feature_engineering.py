import logging
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

class BaseFeatureEngineer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame)-> Dict[str, pd.Series]:
        pass

class EMAFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, ema_span: List[int] = None)->None:
        self._spans: List[int] = ema_span if ema_span is not None else [20, 50, 100, 200]
    
    def transform(self, df: pd.DataFrame)-> Dict[str, pd.Series]:
        if "Close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")
        close = df["Close"]
        indicators = {}
        for s in self._spans:
            indicators[f"ema_{s}"] = close.ewm(span=s, adjust=False).mean()
            logger.debug(f"computed ema for ema_{s}")
        
        logger.info(f"EMAEngineering: {list(indicators.keys())}")
        return indicators

class FeatureEngineer:
    def __init__(self, strategy: BaseFeatureEngineer)-> None:
        logger.info(f"Setting the strategy for feature engineering: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def set_strategy(self, strategy: BaseFeatureEngineer)-> None:
        logger.info(f"Changing the feature engineering strategy: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def transform(self, df: pd.DataFrame)-> Dict[str, pd.Series]:
        logger.info("Performing Feature Engineering on the DataSet using the selected Strategy")
        return self._strategy.transform(df)