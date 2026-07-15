"""
Feature engineering strategies for OHLCV price data, following the Strategy pattern.

AVAILABLE STRATEGIES (all inherit from BaseFeatureEngineer):
    1. EMAFeatureEngineer          — Exponential Moving Averages over configurable spans.
    2. RSIFeatureEngineer          — Relative Strength Index (Wilder-smoothed).
    3. MACDFeatureEngineer         — MACD line, signal line, and histogram (EMA-based).
    4. ATRFeatureEngineer          — Average True Range (Wilder-smoothed).
    5. BollingerBandsFeatureEngineer — Upper/middle/lower bands (population std, ddof=0).
    6. OBVFeatureEngineer          — On-Balance Volume.
    7. DailyReturnFeatureEngineer  — Percentage close-to-close daily return.
    8. PriceRelationshipFeatureEngineer — Intraday High-Low range and Close-Open change.

COMPOSITION:
    CompositeFeatureEngineer — runs a list of strategies against the same input
        DataFrame and merges their outputs into one DataFrame, guarding against
        duplicate feature names.
    FeatureEngineer           — thin public-facing wrapper around
        CompositeFeatureEngineer, for use by the shared prediction pipeline.
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
    def _validate_dataframe(df: pd.DataFrame, required_columns: set[str]) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given input is not a DataFrame.")
        if df.empty:
            raise ValueError("Given DataFrame is empty.")

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing_columns)}")

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

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
        self._spans: list[int] = (list(ema_span) if ema_span is not None else [10, 20, 50, 100, 200])

        if not self._spans:
            raise ValueError("ema_spans cannot be empty.")

        for span in self._spans:
            self._validate_positive_int(span, "EMA span")

        if len(set(self._spans)) != len(self._spans):
            raise ValueError("EMA spans must be unique.")
        
    
    def transform(self, df: pd.DataFrame)-> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close"})
        
        logger.info(f"Calculation EMA's = {self._spans} for the DataSet")
        close = df["Close"]
        indicators: dict[str, pd.Series] = {}
        for s in self._spans:

            indicators[f"EMA{s}"] = close.ewm(span=s, adjust=False).mean()
            logger.debug("computed ema for EMA%s",s)
        
        logger.info("EMAEngineering: %s",list(indicators.keys()))
        return indicators

class RSIFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, period: int = 14) -> None:
        self._validate_positive_int(period, "RSI period")
        self._period: int = period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close"})
        if len(df) <= self._period:
            raise ValueError(f"DataFrame must contain more than {self._period} rows to compute RSI{self._period}.")

        logger.info("Calculating RSI feature for period=%d", self._period)

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
        logger.info("RSI feature calculated successfully: %s", list(indicators.keys()))
        return indicators

class MACDFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> None:
        periods = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
        for name, period in periods.items():
            self._validate_positive_int(period, name)

        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period.")
        self._fast_period: int = fast_period
        self._slow_period: int = slow_period
        self._signal_period: int = signal_period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close"})

        logger.info("Calculating MACD features with fast_period=%d, slow_period=%d, signal_period=%d", self._fast_period, self._slow_period, self._signal_period)

        close = df["Close"]
        fast_ema = close.ewm(span=self._fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self._slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=self._signal_period, adjust=False).mean()
        macd_histogram = macd - macd_signal

        indicators: dict[str, pd.Series] = {"MACD": macd, "MACD_Signal": macd_signal, "MACD_Histogram": macd_histogram}

        logger.debug("Computed MACD features using fast=%s, slow=%s, signal=%s", self._fast_period, self._slow_period, self._signal_period)
        logger.info("MACD features calculated successfully: %s", list(indicators.keys()))
        return indicators

class ATRFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, period: int = 14) -> None:
        self._validate_positive_int(period, "ATR period")
        self._period: int = period

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"High", "Low", "Close"})
        if len(df) < self._period:
            raise ValueError(f"DataFrame must contain at least {self._period} rows to compute ATR{self._period}.")

        logger.info("Calculating ATR feature for period=%d", self._period)

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
        logger.info("ATR feature calculated successfully: %s", list(indicators.keys()))
        return indicators

class BollingerBandsFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, period: int = 20, std_multiplier: float = 2.0) -> None:
        self._validate_positive_int(period, "Bollinger Bands period")

        if (not isinstance(std_multiplier, (int, float)) or isinstance(std_multiplier, bool)):
            raise TypeError("Standard deviation multiplier must be a number.")

        if std_multiplier <= 0:
            raise ValueError("Standard deviation multiplier must be positive.")

        self._period: int = period
        self._std_multiplier: float = float(std_multiplier)

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close"})

        if len(df) < self._period:
            raise ValueError(f"DataFrame must contain at least {self._period} rows to compute Bollinger Bands.")

        logger.info("Calculating Bollinger Bands features for period=%d and std_multiplier=%s", self._period, self._std_multiplier)

        close = df["Close"]

        middle_band = close.rolling(window=self._period, min_periods=self._period).mean()

        rolling_std = close.rolling(window=self._period, min_periods=self._period).std(ddof=0)

        upper_band = middle_band + self._std_multiplier * rolling_std

        lower_band = middle_band - self._std_multiplier * rolling_std

        indicators: dict[str, pd.Series] = { "Bollinger_Upper": upper_band, "Bollinger_Middle": middle_band, "Bollinger_Lower": lower_band, }

        logger.debug("Computed Bollinger Bands using period=%s, " "std_multiplier=%s", self._period, self._std_multiplier)
        logger.info("Bollinger Bands features calculated successfully: %s", list(indicators.keys()))

        return indicators

class OBVFeatureEngineer(BaseFeatureEngineer):
    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close", "Volume"})

        logger.info("Calculating OBV feature")

        close = df["Close"]
        volume = df["Volume"]
        price_change = close.diff()
        direction = np.sign(price_change).fillna(0)
        signed_volume = volume * direction
        obv = signed_volume.cumsum()
        indicators: dict[str, pd.Series] = {"OBV": obv}

        logger.debug("Computed OBV feature")
        logger.info("OBV feature calculated successfully: %s", list(indicators.keys()))

        return indicators

class DailyReturnFeatureEngineer(BaseFeatureEngineer):
    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Close"})

        logger.info("Calculating Daily Return feature")

        close = df["Close"]
        daily_return = close.pct_change(fill_method=None)
        indicators: dict[str, pd.Series] = {"Daily_Return": daily_return}

        logger.debug("Computed Daily Return feature.")
        logger.info("Daily Return feature calculated successfully: %s", list(indicators.keys()))

        return indicators

class PriceRelationshipFeatureEngineer(BaseFeatureEngineer):
    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        self._validate_dataframe(df, {"Open", "High", "Low", "Close"})

        logger.info("Calculating price relationship features")

        high_low = df["High"] - df["Low"]
        open_close = df["Close"] - df["Open"]
        indicators: dict[str, pd.Series] = {"High_Low": high_low, "Open_Close": open_close}

        logger.debug("PriceRelationshipEngineering computed: %s", list(indicators.keys()))
        logger.info("Price relationship features calculated successfully: %s", list(indicators.keys()))

        return indicators

class CompositeFeatureEngineer:
    def __init__(self, strategies: list[BaseFeatureEngineer] | None = None) -> None:
        self._strategies: list[BaseFeatureEngineer] = (
            list(strategies) if strategies is not None else 
            [
                EMAFeatureEngineer(),
                RSIFeatureEngineer(),
                MACDFeatureEngineer(),
                ATRFeatureEngineer(),
                BollingerBandsFeatureEngineer(),
                OBVFeatureEngineer(),
                DailyReturnFeatureEngineer(),
                PriceRelationshipFeatureEngineer(),
            ]
        )

        if not self._strategies:
            raise ValueError("Feature engineering strategies cannot be empty.")

        for strategy in self._strategies:
            if not isinstance(strategy, BaseFeatureEngineer):
                raise TypeError("All strategies must inherit from BaseFeatureEngineer.")
        
        logger.info(f"Setting the strategy for Feature Engineering: {strategy.__class__.__name__}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        BaseFeatureEngineer._validate_dataframe(df, {"Open", "High", "Low", "Close", "Volume"})

        logger.info(f"Starting feature engineering using {len(self._strategies)} strategies")

        engineered_df = df.copy()

        for strategy in self._strategies:
            features = strategy.transform(df)

            if not isinstance(features, dict):
                raise TypeError(f"{strategy.__class__.__name__}.transform() must return a dictionary.")

            duplicate_features = (set(features) & set(engineered_df.columns))
            if duplicate_features:
                raise ValueError(f"Duplicate feature names detected: {sorted(duplicate_features)}")

            for feature_name, feature_series in features.items():
                if not isinstance(feature_name, str):
                    raise TypeError("Feature names must be strings.")
                if not isinstance(feature_series, pd.Series):
                    raise TypeError(f"Feature '{feature_name}' must be a pandas Series.")
                if not feature_series.index.equals(df.index):
                    raise ValueError(f"Feature '{feature_name}' index does not match the input DataFrame index.")                

                engineered_df[feature_name] = feature_series

        logger.info(f"Feature engineering completed successfully: {engineered_df.shape[1] - df.shape[1]} features added, total columns={engineered_df.shape[1]}")

        return engineered_df
    
class FeatureEngineer:
    def __init__(self, engineer: CompositeFeatureEngineer | None = None) -> None:
        if (engineer is not None and not isinstance(engineer, CompositeFeatureEngineer)):
            raise TypeError("engineer must be a CompositeFeatureEngineer.")

        self._engineer: CompositeFeatureEngineer = (engineer if engineer is not None else CompositeFeatureEngineer())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming the Dataset to add 17 More Features")
        return self._engineer.transform(df)