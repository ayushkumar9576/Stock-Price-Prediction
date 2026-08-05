import logging
from abc import ABC, abstractmethod
from pathlib import Path
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from utils.console import starting

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metrics_path: str, pred_path: str)-> None:
        pass

    @abstractmethod
    def plot(self, y_true: ArrayLike, y_pred: ArrayLike, *, dates: ArrayLike | None = None, title: str = "Stock Price Prediction", save_path: str | Path = "results/prediction.png")-> None:
        pass

class RegressionEvaluator(BaseEvaluator):
    def __init__(self):
        logger.info("RegressionEvaluator initialized.")

    @staticmethod
    def _validate_array(y_true: ArrayLike, y_pred: ArrayLike)-> tuple[np.ndarray, np.ndarray]:
        try:
            y_true = np.asarray(y_true, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError("'y_true' must be array-like and convertible to a numeric numpy array.") from e

        try:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError("'y_pred' must be array-like and convertible to a numeric numpy array.") from e

        if y_true.size == 0:
            raise ValueError("'y_true' cannot be empty.")
        if y_pred.size == 0:
            raise ValueError("'y_pred' cannot be empty.")

        if y_true.ndim not in (1, 2):
            raise ValueError("'y_true' must be a 1D array or a 2D single-column array.")
        if y_pred.ndim not in (1, 2):
            raise ValueError("'y_pred' must be a 1D array or a 2D single-column array.")

        if y_true.ndim == 2 and y_true.shape[1] != 1:
            raise ValueError("A 2D 'y_true' array must contain exactly one column.")
        if y_pred.ndim == 2 and y_pred.shape[1] != 1:
            raise ValueError("A 2D 'y_pred' array must contain exactly one column.")

        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("'y_true' and 'y_pred' must contain the same number of samples.")

        if not np.all(np.isfinite(y_true)):
            raise ValueError("'y_true' must contain only finite values.")
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("'y_pred' must contain only finite values.")

        return y_true, y_pred

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metrics_path: str = "results/metrics.json", pred_path: str = "results/predictions.txt")-> None:
        y_true, y_pred = self._validate_array(y_true, y_pred)

        mse = mean_squared_error(y_true, y_pred)
        metrices = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)) ,
            "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred)*100),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrices, file, indent=4)

        with open(pred_path, "w", encoding="utf-8") as file:
            file.write(f"{'Index':<8}{'Actual':>15}{'Predicted':>15}{'Error':>15}\n")
            file.write("-" * 53 + "\n")
            for idx, (actual, predicted) in enumerate(zip(y_true, y_pred)):
                actual = float(actual)
                predicted = float(predicted)
                error = predicted - actual
                file.write(f"{idx:<8} {actual:>15.4f} {predicted:>15.4f} {error:>15.4f}\n")

        logger.info("Regression metrics: %s", metrices)
        logger.info("Evaluation artifacts saved to '%s' and '%s'.",metrics_path, pred_path)

    def plot(self, y_true: ArrayLike, y_pred: ArrayLike, *, dates: ArrayLike | None = None, title: str = "Stock Price Prediction", save_path: str | Path = "results/prediction.png")-> None:
        y_true, y_pred = self._validate_array(y_true, y_pred)

        save_path = Path(save_path).expanduser()
        if save_path.suffix.lower() != ".png":
            raise ValueError("'save_path' must have a '.png' extension.")

        if dates is not None:
            try:
                dates = np.asarray(dates)
            except (TypeError, ValueError) as e:
                raise TypeError("'dates' must be array-like.") from e

            if dates.ndim not in (1, 2):
                raise ValueError("'dates' must be a 1D array or a 2D single-column array.")
            if dates.ndim == 2 and dates.shape[1] != 1:
                raise ValueError("A 2D 'dates' array must contain exactly one column.")

            dates = dates.reshape(-1)

            if dates.shape[0] != y_true.shape[0]:
                raise ValueError("'dates' must have the same length as 'y_true' and 'y_pred'.")

            x_axis = dates
        else:
            x_axis = np.arange(len(y_true))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        figure = plt.figure(figsize=(14, 6))
        try:
            plt.plot(x_axis, y_true, label="Actual Price", linewidth=2)
            plt.plot(x_axis, y_pred, linestyle="--", linewidth=2, label="Predicted Price")
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(figure)

        logger.info(f"Prediction plot saved successfully to '{save_path}'.")

class ModelEvaluator:
    def __init__(self, strategy: BaseEvaluator)-> None:
        self.name = "Model Evaluator"
        self.start_time = starting(self.name)

        self._validate_strategy(strategy)
        logger.info(f"Setting the strategy for Model Evaluator: {strategy.__class__.__name__}")
        self._strategy = strategy

    @staticmethod
    def _validate_strategy(strategy: BaseEvaluator)-> None:
        if not isinstance(strategy, BaseEvaluator):
            raise TypeError(f"Expected a BaseEvaluator, got {type(strategy)}")

    @property
    def strategy(self) -> BaseEvaluator:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseEvaluator) -> None:
        self._validate_strategy(strategy)
        logger.info(f"Evaluator — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseEvaluator)-> None:
        self._validate_strategy(strategy)

        logger.info(f"Changing the strategy for Model Evaluator: {strategy.__class__.__name__}")
        self._strategy = strategy

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metrics_path: str = "results/metrics.json", pred_path: str = "results/predictions.txt") -> None:
        logger.info("Evaluating the model with the selected Strategy")
        return self._strategy.evaluate(y_true, y_pred, metrics_path, pred_path)

    def plot(self, y_true: ArrayLike, y_pred: ArrayLike, *, dates: ArrayLike | None = None, title: str = "Stock Price Prediction", save_path: str | Path = "results/prediction.png") -> None:
        logger.info("Creating and Saving the Prediction graph")
        self._strategy.plot(y_true, y_pred, dates= dates, title= title, save_path= save_path)
