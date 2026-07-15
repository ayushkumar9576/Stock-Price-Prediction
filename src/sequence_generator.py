from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

class BaseSequencer(ABC):
    @property
    @abstractmethod
    def lookback(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _validate_input(X: NDArray[np.number], y: NDArray[np.number], X_name: str = "X", y_name: str = "y")-> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{X_name} must be a NumPy array.")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"{y_name} must be a NumPy array.")

        if X.ndim != 2:
            raise ValueError(f"{X_name} must be a 2D array with shape (n_samples, n_features).")

        if y.ndim not in (1, 2):
            raise ValueError(f"{y_name} must be a 1D array with shape (n_samples,) or a 2D single-column array with shape (n_samples, 1).")

        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"A 2D {y_name} array must contain exactly one target column.")

        if X.shape[0] == 0:
            raise ValueError(f"{X_name} cannot be empty.")

        if y.shape[0] == 0:
            raise ValueError(f"{y_name} cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError(f"{X_name} must contain at least one feature.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{X_name} and {y_name} must contain the same number of samples. Received {X_name}={X.shape[0]} and {y_name}={y.shape[0]}.")

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"{X_name} must contain numeric values.")

        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"{y_name} must contain numeric values.")

        if not np.all(np.isfinite(X)):
            raise ValueError(f"{X_name} must contain only finite values.")

        if not np.all(np.isfinite(y)):
            raise ValueError(f"{y_name} must contain only finite values.")
        
    @staticmethod
    def _validate_feature_consistency(X_train: NDArray[np.number], X_test: NDArray[np.number]) -> None:
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"X_train and X_test must contain the same number of features. \nReceived X_train={X_train.shape[1]} features and \nX_test={X_test.shape[1]} features.")
    
    @abstractmethod
    def generate_sequences(self, X: NDArray[np.number], y: NDArray[np.number]) -> tuple[NDArray[np.number], NDArray[np.number]]:
        raise NotImplementedError
    
    @abstractmethod
    def generate_train_test_sequences(self, X_train: NDArray[np.number], y_train: NDArray[np.number], X_test: NDArray[np.number], y_test: NDArray[np.number]) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number], NDArray[np.number]]:  
        raise NotImplementedError
    
class SlidingWindowSequencer(BaseSequencer):
    def __init__(self, lookback: int = 100) -> None:
        self._validate_lookback(lookback)
        self._lookback = lookback
    
    @property
    def lookback(self) -> int:
        return self._lookback
    
    @staticmethod
    def _validate_lookback(lookback: int) -> None:
        if isinstance(lookback, bool) or not isinstance(lookback, int):
            raise TypeError("lookback must be an integer.")
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
    
    def generate_sequences(self, X: NDArray[np.number], y: NDArray[np.number]) -> tuple[NDArray[np.number], NDArray[np.number]]:
        self._validate_input(X, y)

        n_sample = X.shape[0]

        if n_sample <=self._lookback:
            raise ValueError(F"The number of sample must be greater then lookback={self._lookback}. Received {n_sample} samples.")
        
        y_flat = y.reshape(-1)
        n_sequence = n_sample - self._lookback
        n_features = X.shape[1]

        X_sequence = np.empty((n_sequence, self._lookback, n_features), dtype=X.dtype)
        y_sequence = np.empty(n_sequence, dtype= y_flat.dtype)

        for s_idx in range(n_sequence):
            target_idx = s_idx + self._lookback
            X_sequence[s_idx] = X[s_idx:target_idx]
            y_sequence[s_idx] = y_flat[target_idx]

        logger.info(f"Sequences generated successfully: X_shape={X_sequence.shape}, y_shape={y_sequence.shape}")

        return X_sequence, y_sequence
    
    def generate_train_test_sequences(self, X_train: NDArray[np.number], y_train: NDArray[np.number], X_test: NDArray[np.number], y_test: NDArray[np.number]) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
        self._validate_input(X_train, y_train, X_name="X_train", y_name="y_train")
        self._validate_input(X_test, y_test, X_name="X_test", y_name="y_test")
        self._validate_feature_consistency(X_train, X_test)

        if X_train.shape[0] <=self._lookback:
            raise ValueError(f"X_train must contain more rows then lookbacl={self._lookback}\nTo generate at least 1 training sequence and provide historical context for the test case.\nRecieved {X_train.shape[0]} Rows")

        X_train_sequence, y_train_sequece = self.generate_sequences(X_train, y_train)

        X_context = np.concatenate((X_train[-self._lookback:], X_test), axis=0)
        y_context = np.concatenate((y_train.reshape(-1)[-self._lookback:], y_test.reshape(-1)), axis=0)

        X_test_sequence, y_test_sequence = self.generate_sequences(X_context, y_context)
        
        logger.info(f"Training and testing sequences generated successfully: X_train_shape={X_train_sequence.shape}, y_train_shape={y_train_sequece.shape}, X_test_shape={X_test_sequence.shape}, y_test_shape={y_test_sequence.shape}")

        return X_train_sequence, y_train_sequece, X_test_sequence, y_test_sequence
    

class Sequencer:
    def __init__(self, strategy: BaseSequencer)-> None:
        self._validate_strategy(strategy)
        logger.info(f"Setting the strategy for Sequence Generator: {strategy.__class__.__name__}")
        self._strategy = strategy

    @staticmethod
    def _validate_strategy(strategy: BaseSequencer)-> None:
        if not isinstance(strategy, BaseSequencer):
            raise TypeError(f"Expected a BaseSequencer, got {type(strategy)}")
    
    @property
    def strategy(self) -> BaseSequencer:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseSequencer) -> None:
        self._validate_strategy(strategy)
        logger.info(f"Sequencer — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseSequencer)-> None:
        self._validate_strategy(strategy)

        logger.info(f"Changing the strategy for Sequence Generator: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def generate_sequence(self, X_train: NDArray[np.number], y_train: NDArray[np.number])-> tuple[NDArray[np.number], NDArray[np.number]]:
        logger.info("Generating sequence for the Training Data")
        return self._strategy.generate_sequences(X_train, y_train)
    
    def generate_train_test_sequence(self, X_train: NDArray[np.number], y_train: NDArray[np.number], X_test: NDArray[np.number], y_test: NDArray[np.number])-> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
        logger.info("Generating sequences for the Training and Testing data")
        return self._strategy.generate_train_test_sequences(X_train, y_train, X_test, y_test)