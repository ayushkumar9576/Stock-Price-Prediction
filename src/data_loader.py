import os
import logging
from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, ticker: str, start: str, end: str)-> pd.DataFrame:
        pass

class YFinanceLoader(BaseDataLoader):

    def __init__(self, raw_dir: str)-> None:
        self._raw_dir = raw_dir
    
    def load(self, ticker: str, start: str, end: str)-> pd.DataFrame:
        logger.info("Downloading and caching the stock price data from Yahoo finance")
        os.makedirs(self._raw_dir, exist_ok=True)
        cache_path = os.path.join(self._raw_dir, f"{ticker}.csv")

        if os.path.exists(cache_path):
            logger.info(f"[{ticker}] Loading the data from the cache file")
            return pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        
        logger.info(f"[{ticker}] downloading the data from the Yahoo Finance between ({start} to {end})")

        df = yf.download(ticker, start= start,end= end, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}")
        
        df.to_csv(cache_path)
        logger.info(f"Downloaded the data for {ticker} between {start} and {end}")
        return df

class CSVLoader(BaseDataLoader):
    def __init__(self, raw_dir: str)-> None:
        self._raw_dir = raw_dir
    
    def load(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        logger.info("Loading the stock data from a pre-loaded CSV")
        csv_path = os.path.join(self._raw_dir, f"{ticker}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No cached CSV found at {csv_path}. Run with YFinanceLoader first to download and cache the data.")

        logger.info(f"[{ticker}] Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = df.loc[start:end]

        if df.empty:
            raise ValueError(f"CSV for '{ticker}' has no data between {start} and {end}.")

        logger.info(f"[{ticker}] Loaded {len(df)} rows from CSV.")
        return df

class DataLoader:
    def __init__(self, strategy: BaseDataLoader)-> None:
        logger.info(f"Setting the strategy for data loading: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def set_strategy(self, strategy: BaseDataLoader)-> None:
        logger.info(f"Changing the strategy for loading the data: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def load(self, ticker: str, start: str, end: str)-> pd.DataFrame:
        logger.info("Loading the data using the selected strategy")
        return self._strategy.load(ticker, start, end)