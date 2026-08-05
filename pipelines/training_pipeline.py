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

_LOADER_STRATEGIES = {
    "YFinanceLoader": YFinanceLoader,
    "CSVLoader": CSVLoader
}

_PREPROCESSOR_STRATEGIES = {
    "StandardDataScaler": StandardDataScaler
}

_FEATURE_ENGINEERING_STRATEGIES = {
    "CompositeFeatureEngineer": CompositeFeatureEngineer
}

_MISSING_VALUE_STRATEGIES = {
    "DropMissingValues": DropMissingValues,
    "TimeSeriesImputer": TimeSeriesImputer,
    "MeanImputer": MeanImputer
}

_MODEL_STRATEGIES = {
    "LSTMModel": LSTMModel
}

_EVALUATOR_STRATEGIES = {
    "RegressionEvaluator": RegressionEvaluator
}

_OUTLIER_STRATEGIES = {
    "IQROutlierDetector": IQROutlierDetector,
    "ZScoreOutlierDetector": ZScoreOutlierDetector
}

_SEQUENCE_STRATEGIES = {
    "SlidingWindowSequencer": SlidingWindowSequencer
}

def train_pipeline():
    config = get_config()

    loader_cfg = config["data_loader"]
    preprocess_cfg = config["data_splitter_and_processing"]
    feature_cfg = config["feature_engineering"]
    missing_cfg = config["missing_value"]
    model_cfg = config["model"]
    evaluation_cfg = config["model_evaluation"]
    outlier_cfg = config["outlier_detection"]
    sequence_cfg = config["sequence_generator"]
    path_cfg = config["path"]
    pipeline_cfg = config["pipelines"]

    data_loader = DataLoader(_LOADER_STRATEGIES[loader_cfg["strategy"]](resolve_path(path_cfg["raw_data"])))
    df = data_loader.load(loader_cfg["ticker"], loader_cfg["start_date"], loader_cfg["end_date"])

    missing_value = MissingValueHandler(_MISSING_VALUE_STRATEGIES[missing_cfg["initial"]["strategy"]](columns=missing_cfg["initial"]["columns"]))
    df = missing_value.handle_missing_value(df)

    feature_engineering = FeatureEngineer(_FEATURE_ENGINEERING_STRATEGIES[feature_cfg["strategy"]](
        [EMAFeatureEngineer(ema_span=feature_cfg["ema"]["spans"]), 
        RSIFeatureEngineer(period=feature_cfg["rsi"]["period"]),
        MACDFeatureEngineer(fast_period=feature_cfg["macd"]["fast_period"], slow_period=feature_cfg["macd"]["slow_period"], signal_period=feature_cfg["macd"]["signal_period"]), 
        ATRFeatureEngineer(period=feature_cfg["atr"]["period"]), 
        BollingerBandsFeatureEngineer(period=feature_cfg["bollinger_bands"]["period"], std_multiplier=feature_cfg["bollinger_bands"]["std_multiplier"]), 
        OBVFeatureEngineer(), 
        DailyReturnFeatureEngineer(), 
        PriceRelationshipFeatureEngineer()]
        )
    )    
    df = feature_engineering.transform(df)

    missing_value.set_strategy(_MISSING_VALUE_STRATEGIES[missing_cfg["after_feature_engineering"]["strategy"]](axis=missing_cfg["after_feature_engineering"]["axis"], thresh=missing_cfg["after_feature_engineering"]["thresh"]))
    df = missing_value.handle_missing_value(df)

    raw_close = df["Close"].values.copy()
    target_column = config["target"]["column"]
    df[target_column] = df["Close"].pct_change().shift(-1)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    split_idx = int(len(df) * preprocess_cfg["train_split_ratio"])
    df_train_portion = df.iloc[:split_idx].copy()
    df_test_portion  = df.iloc[split_idx:].copy()

    outlier_detection = OutlierDetection(_OUTLIER_STRATEGIES[outlier_cfg["strategy"]](columns=outlier_cfg["columns"], factor=outlier_cfg["factor"], action=outlier_cfg["action"]))
    df_train_portion = outlier_detection.detect_and_handle(df_train_portion)

    df = pd.concat([df_train_portion, df_test_portion])

    feature_cols = [c for c in df.columns if c != target_column]
    splitter_and_preprocessing = DataPreprocessorAndSplitting(_PREPROCESSOR_STRATEGIES[preprocess_cfg["strategy"]]())
    X_train, X_test, y_train, y_test = splitter_and_preprocessing.split(df, feature_cols=feature_cols, target_column=target_column)

    splitter_and_preprocessing.fit(X_train, y_train)
    (X_train, y_train), (X_test, y_test) = splitter_and_preprocessing.transform(X_train, y_train), splitter_and_preprocessing.transform(X_test, y_test)
    completion(splitter_and_preprocessing.name, splitter_and_preprocessing.start_time)


    sequence_generator = Sequencer(_SEQUENCE_STRATEGIES[sequence_cfg["strategy"]](lookback=sequence_cfg["lookback"]))
    X_train_sequence, y_train_sequence, X_test_sequence, y_test_sequence = sequence_generator.generate_train_test_sequence(X_train, y_train, X_test, y_test)

    model = Model(_MODEL_STRATEGIES[model_cfg["strategy"]](
        units=model_cfg["units"], 
        dropout_rates=model_cfg["dropout_rates"], 
        dense_units=model_cfg["dense_units"], 
        learning_rate=model_cfg["learning_rate"], 
        clipnorm=model_cfg["clipnorm"], 
        epochs=model_cfg["epochs"], 
        batch_size=model_cfg["batch_size"], 
        validation_split=model_cfg["validation_split"], 
        early_stopping_patience=model_cfg["early_stopping"]["patience"], 
        early_stopping_min_delta=model_cfg["early_stopping"]["min_delta"], 
        reduce_lr_patience=model_cfg["reduce_lr"]["patience"], 
        reduce_lr_factor=model_cfg["reduce_lr"]["factor"], 
        min_learning_rate=model_cfg["reduce_lr"]["min_learning_rate"], 
        checkpoint_path=resolve_path(path_cfg["model"]))
    )
    if pipeline_cfg["train_model"]:
        model.build(input_shape=(X_train_sequence.shape[1], X_train_sequence.shape[2]))
        model.train(X_train_sequence, y_train_sequence)
    else:
        model.load(resolve_path(path_cfg["model"]))
    y_pred_scaled = model.predict(X_test_sequence)
    completion(model.name, model.start_time)

    pred_returns = splitter_and_preprocessing.inverse_transform(y_pred_scaled).flatten()
    n_preds = len(pred_returns)
    test_base_prices = raw_close[split_idx : split_idx + n_preds]
    test_actual_prices = raw_close[split_idx + 1 : split_idx + 1 + n_preds]
    y_pred_close = test_base_prices * (1.0 + pred_returns)
    y_true_close = test_actual_prices
    evaluator = ModelEvaluator(_EVALUATOR_STRATEGIES[evaluation_cfg["strategy"]]())
    evaluator.evaluate(y_true=y_true_close, y_pred=y_pred_close, metrics_path=resolve_path(path_cfg["metrics"]), pred_path=resolve_path(path_cfg["prediction"]))
    evaluator.plot(y_true=y_true_close, y_pred=y_pred_close, save_path=resolve_path(path_cfg["plot"]))
    completion(evaluator.name, evaluator.start_time)


if __name__ == "__main__":
    train_pipeline()