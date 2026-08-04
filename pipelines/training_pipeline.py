"""
pipeline/training_pipeline.py
-----------------------------
Full end-to-end training pipeline.

Called exclusively by train.py (or main.py). Reads config.yaml once,
instantiates all required components and strategies, and executes the
complete model training workflow.

Stages:
  1.  Load Configuration          → get_config()
  2.  Data Loading                → DataLoader.load()
  3.  Missing Value Handling      → MissingValueHandler.handle()
  4.  Outlier Detection           → OutlierDetector.detect_and_handle()
  5.  Feature Engineering         → FeatureEngineer.transform()
  6.  Data Preprocessing          → Preprocessor.fit_transform()
  7.  Sequence Generation         → SequenceGenerator.generate()
  8.  Model Building              → ModelBuilder.build()
  9.  Model Training              → ModelTrainer.train()
 10.  Model Evaluation            → ModelEvaluator.evaluate()
 11.  Prediction Visualization    → ModelEvaluator.plot()
 12.  Save Trained Model          → ModelTrainer.save()
"""
import logging
from src.config_loader import get_config, resolve_path
from src.data_loader import DataLoader, YFinanceLoader, CSVLoader
from src.missing_value_handler import MissingValueHandler, DropMissingValues, MeanImputer, TimeSeriesImputer
from src.outlier_detector import OutlierDetection, IQROutlierDetector, ZScoreOutlierDetector
from src.feature_engineering import  FeatureEngineer, CompositeFeatureEngineer, EMAFeatureEngineer, RSIFeatureEngineer, MACDFeatureEngineer, ATRFeatureEngineer, BollingerBandsFeatureEngineer, OBVFeatureEngineer, DailyReturnFeatureEngineer, PriceRelationshipFeatureEngineer
from src.data_splitter_and_processing import DataPreprocessorAndSplitting, StandardDataScaler
from src.sequence_generator import Sequencer, SlidingWindowSequencer
from src.model_builder import Model, LSTMModel
from src.model_evaluator import ModelEvaluator, RegressionEvaluator
from rich import print
import time
import pandas as pd
from utils.console import starting, completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def train_pipeline():

    data_loader = DataLoader(YFinanceLoader("data/raw"))
    df = data_loader.load("TSLA", "2010-06-29", "2026-07-30")

    missing_value = MissingValueHandler(TimeSeriesImputer())
    df = missing_value.handle_missing_value(df)

    feature_engineering = FeatureEngineer(CompositeFeatureEngineer([EMAFeatureEngineer(), RSIFeatureEngineer(), MACDFeatureEngineer(), ATRFeatureEngineer(), BollingerBandsFeatureEngineer(), OBVFeatureEngineer(), DailyReturnFeatureEngineer(), PriceRelationshipFeatureEngineer()]))    
    df = feature_engineering.transform(df)

    missing_value.set_strategy(DropMissingValues())
    df = missing_value.handle_missing_value(df)

    df["Target_Return"] = df["Close"].pct_change().shift(-1)
    df = df.dropna(subset=["Target_Return"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.70)
    df_train_portion = df.iloc[:split_idx].copy()
    df_test_portion  = df.iloc[split_idx:].copy()

    outlier_detection = OutlierDetection(IQROutlierDetector(columns=["Volume"]))
    df_train_portion = outlier_detection.detect_and_handle(df_train_portion)

    df = pd.concat([df_train_portion, df_test_portion])

    feature_cols = [c for c in df.columns if c != "Target_Return"]
    splitter_and_preprocessing = DataPreprocessorAndSplitting(StandardDataScaler())
    X_train, X_test, y_train, y_test = splitter_and_preprocessing.split(df, feature_cols=feature_cols, target_column="Target_Return")
    test_close_raw = df["Close"].iloc[split_idx:].values

    splitter_and_preprocessing.fit(X_train, y_train)
    (X_train, y_train), (X_test, y_test) = splitter_and_preprocessing.transform(X_train, y_train), splitter_and_preprocessing.transform(X_test, y_test)
    completion(splitter_and_preprocessing.name, splitter_and_preprocessing.start_time)


    sequence_generator = Sequencer(SlidingWindowSequencer())
    X_train_sequence, y_train_sequence, X_test_sequence, y_test_sequence = sequence_generator.generate_train_test_sequence(X_train, y_train, X_test, y_test)

    model = Model(LSTMModel(units=[256, 128], dropout_rates=[0.05, 0.05], dense_units=64, learning_rate=0.001, clipnorm=1, epochs=200, batch_size=16, validation_split=0.1, early_stopping_patience=25, early_stopping_min_delta=1e-5, reduce_lr_patience=3, reduce_lr_factor=0.5, min_learning_rate=1e-6, checkpoint_path="model/lstm_MODEL.keras"))
    if _MODEL:
        model.build(input_shape=(X_train_sequence.shape[1], X_train_sequence.shape[2]))
        model.train(X_train_sequence, y_train_sequence)
    else:
        model.load("model/lstm_MODEL.keras")
    y_pred_scaled = model.predict(X_test_sequence)
    completion(model.name, model.start_time)

    pred_returns = splitter_and_preprocessing.inverse_transform(y_pred_scaled).flatten()
    n_preds = len(pred_returns)
    test_base_prices = df["Close"].iloc[split_idx - 1 : split_idx - 1 + n_preds].values
    test_actual_prices = df["Close"].iloc[split_idx : split_idx + n_preds].values
    y_pred_close = test_base_prices * (1.0 + pred_returns)
    y_true_close = test_actual_prices
    evaluator = ModelEvaluator(RegressionEvaluator())
    evaluator.evaluate(y_true_close, y_pred_close)
    evaluator.plot(y_true_close, y_pred_close)


if __name__ == "__main__":
    train_pipeline()