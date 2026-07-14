"""
AVAILABLE STRATEGIES:
    1. EMAFeatureEngineer — Generates Exponential Moving Average (EMA) features over specified window spans.
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BaseFeatureEngineer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame)-> dict[str, pd.Series]:
        raise NotImplementedError

    @staticmethod
    def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
        values = series.to_numpy(dtype="float64")
        n = len(values)
        smoothed = np.full(n, np.nan, dtype="float64")
        consecutive_valid = 0
        seed_index = None
        for i, value in enumerate(values):
            if np.isnan(value):
                consecutive_valid = 0
            else:
                consecutive_valid += 1
            if consecutive_valid == period:
                seed_index = i
                break
        if seed_index is None:
            return pd.Series(smoothed, index=series.index)
        seed_start = seed_index - period + 1
        smoothed[seed_index] = np.mean(values[seed_start : seed_index + 1])
        for i in range(seed_index + 1, n):
            if np.isnan(values[i]):
                smoothed[i] = np.nan
                continue
            if np.isnan(smoothed[i - 1]):
                break
            smoothed[i] = (smoothed[i - 1] * (period - 1) + values[i]) / period
        return pd.Series(smoothed, index=series.index)

class EMAFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, ema_span: list[int] | None = None)->None:
        if ema_span is not None and not isinstance(ema_span, list):
            raise TypeError("ema_span must be a list of integers.")
        self._spans: list[int] = ema_span if ema_span is not None else [10, 20, 50, 100, 200]
        
        if not isinstance(self._spans, list):
            raise TypeError("ema_spans must be a list of integers.")

        if not self._spans:
            raise ValueError("ema_spans cannot be empty.")

        if any(not isinstance(span, int) or isinstance(span, bool) for span in self._spans):
            raise TypeError("All EMA spans must be integers.")

        if any(span <= 0 for span in self._spans):
            raise ValueError("All EMA spans must be positive integers.")

        if len(set(self._spans)) != len(self._spans):
            raise ValueError("EMA spans must be unique.")
        
    
    def transform(self, df: pd.DataFrame)-> dict[str, pd.Series]:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given input is not DataFrame")
        if df.empty:
            raise ValueError("Given DataFrame is empty.")

        if "Close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")
        
        close = df["Close"]
        indicators: dict[str, pd.Series] = {}
        for s in self._spans:

            indicators[f"EMA{s}"] = close.ewm(span=s, adjust=False).mean()
            logger.debug(f"computed ema for EMA{s}")
        
        logger.info(f"EMAEngineering: {list(indicators.keys())}")
        return indicators

class RSIFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, period: int = 14) -> None:
        if not isinstance(period, int) or isinstance(period, bool):
            raise TypeError("RSI period must be an integer.")
        if period <= 0:
            raise ValueError("RSI period must be a positive integer.")
        self._period: int = period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given input is not DataFrame.")
        if df.empty:
            raise ValueError("Given DataFrame is empty.")
        if "Close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column")
        if len(df) <= self._period:
            raise ValueError(f"DataFrame must contain more than {self._period} rows to compute RSI{self._period}.")

        close = df["Close"]
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        average_gain = self._wilder_smoothing(gains, self._period)
        average_loss = self._wilder_smoothing(losses, self._period)

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_strength = average_gain / average_loss
            rsi = 100 - (100 / (1 + relative_strength))

        flat_mask = (average_gain == 0) & (average_loss == 0)
        rsi = rsi.mask(flat_mask, 50.0)

        all_gain_mask = (average_loss == 0) & (average_gain > 0)
        rsi = rsi.mask(all_gain_mask, 100.0)

        feature_name = f"RSI{self._period}"
        indicators: dict[str, pd.Series] = {feature_name: rsi}

        logger.debug(f"Computed RSI feature: {feature_name}")
        logger.info(f"RSIEngineering: {list(indicators.keys())}")
        return indicators

class MACDFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> None:
        periods = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
        for name, period in periods.items():
            if not isinstance(period, int) or isinstance(period, bool):
                raise TypeError(f"{name} must be an integer.")
            if period <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period.")
        self._fast_period: int = fast_period
        self._slow_period: int = slow_period
        self._signal_period: int = signal_period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given input is not DataFrame.")
        if df.empty:
            raise ValueError("Given DataFrame is empty.")
        if "Close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        close = df["Close"]
        fast_ema = close.ewm(span=self._fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self._slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=self._signal_period, adjust=False).mean()
        macd_histogram = macd - macd_signal

        indicators: dict[str, pd.Series] = {"MACD": macd, "MACD_Signal": macd_signal, "MACD_Histogram": macd_histogram}

        logger.debug("Computed MACD features using fast=%s, slow=%s, signal=%s", self._fast_period, self._slow_period, self._signal_period)
        logger.info("MACDEngineering: %s", list(indicators.keys()),)
        return indicators

class ATRFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, period: int = 14) -> None:
        if not isinstance(period, int) or isinstance(period, bool):
            raise TypeError("ATR period must be an integer.")
        if period <= 0:
            raise ValueError("ATR period must be a positive integer.")
        self._period: int = period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given input is not DataFrame.")
        if df.empty:
            raise ValueError("Given DataFrame is empty.")
        required_columns = {"High", "Low", "Close"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing_columns)}")
        if len(df) < self._period:
            raise ValueError(f"DataFrame must contain at least {self._period} rows to compute ATR{self._period}.")

        high = df["High"]
        low = df["Low"]
        previous_close = df["Close"].shift(1)

        high_low = high - low
        high_previous_close = (high - previous_close).abs()
        low_previous_close = (low - previous_close).abs()

        true_range = pd.concat([high_low, high_previous_close, low_previous_close], axis=1).max(axis=1)

        atr = self._wilder_smoothing(true_range, self._period)

        feature_name = f"ATR{self._period}"
        indicators: dict[str, pd.Series] = {feature_name: atr}

        logger.debug("Computed ATR feature: %s", feature_name)
        logger.info("ATREngineering: %s", list(indicators.keys()))
        return indicators

class FeatureEngineer:
    def __init__(self, strategy: BaseFeatureEngineer)-> None:
        if not isinstance(strategy, BaseFeatureEngineer):
            raise TypeError(f"Expected BaseFeatureEngineer, got {type(strategy)}")

        logger.info(f"Setting the strategy for feature engineering: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    @property
    def strategy(self) -> BaseFeatureEngineer:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseFeatureEngineer) -> None:
        if not isinstance(strategy, BaseFeatureEngineer):
            raise TypeError(f"Expected a BaseFeatureEngineer, got {type(strategy)}")
        logger.info(f"FeatureEngineer — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseFeatureEngineer)-> None:
        if not isinstance(strategy, BaseFeatureEngineer):
            raise TypeError(f"Expected BaseFeatureEngineer, got {type(strategy)}")

        logger.info(f"Changing the feature engineering strategy: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def transform(self, df: pd.DataFrame)-> dict[str, pd.Series]:
        logger.info("Performing Feature Engineering on the DataSet using the selected Strategy")
        return self._strategy.transform(df)