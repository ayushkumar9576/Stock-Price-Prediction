import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LSTM
from keras.losses import Huber
from keras.metrics import MeanAbsoluteError
from keras.models import Model as KerasModel, Sequential, load_model as Keras_load_model
from keras.optimizers import Adam

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    @property
    @abstractmethod
    def keras_model(self)-> KerasModel | None:
        pass

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, int]) -> None:
        if not isinstance(input_shape, tuple):
            raise TypeError("Input Shape must be a tuple")
        
        if len(input_shape)!=2:
            raise ValueError("Input shape must only contain 2 dimension: (Lookback, n_features)")
        
        for i in input_shape:
            if not isinstance(i, int) or isinstance(i, bool):
                raise TypeError("Input shape dimension must be Integer")
            if i <=0:
                raise ValueError("Input Shape dimension must be a positive number")
            
    @staticmethod
    def _validate_training_data(X: NDArray[np.number], y:NDArray[np.number], X_name: str = "X", y_name: str = "y")-> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{X_name} must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{y_name} must be a numpy array")

        if X.ndim != 3: 
            raise ValueError(f"{X_name} must be a 3d array with shape (n_sample, Lookback, n_features)")
        if y.ndim not in (1,2):
            raise ValueError(f"{y_name} must either be a 1d array or 2d single-columns array")        

        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"A 2D {y_name} array must contain exactly one target column")
        if y.shape[0] == 0:
            raise ValueError(f"{y_name} cannot be empty.")
        if X.shape[0] == 0:
            raise ValueError(f"{X_name} cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError(f"{X_name} must contain at least one time step.")
        if X.shape[2] == 0:
            raise ValueError(f"{X_name} must contain at least one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{X_name} and {y_name} must contain the same number of samples.\nReceived {X_name}={X.shape[0]} and {y_name}={y.shape[0]}.")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"{X_name} must contain numeric values.")
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"{y_name} must contain numeric values.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{X_name} must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{y_name} must contain only finite values.")
    
    @staticmethod
    def _validate_prediction_data(X: NDArray[np.number])-> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Predicted Data(X) must be a numpy array")

        if X.ndim != 3: 
            raise ValueError(f"Predicted Data(X) must be a 3d array with shape (n_sample, Lookback, n_features)")
        if X.shape[0] == 0:
            raise ValueError(f"Predicted Data(X) cannot be empty.")
        if X.shape[1] == 0:
            raise ValueError(f"Predicted Data(X) must contain at least one time step.")
        if X.shape[2] == 0:
            raise ValueError(f"Predicted Data(X) must contain at least one feature.")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Predicted Data(X) must contain numeric values.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Predicted Data(X) must contain only finite values.")
        
    @abstractmethod
    def build(self, input_shape: tuple[int, int]) -> None:
        pass

    @abstractmethod
    def train(self, X: NDArray[np.number], y: NDArray[np.number]) -> Any:
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.number]) -> NDArray[np.number]:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        pass

class LSTMModel(BaseModel):
    def __init__(self, units: list[int] | None = None, dropout_rates: list[float] | None = None, dense_units: int = 32, learning_rate: float = 0.001, clipnorm: float = 1.0, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.1, early_stopping_patience: int = 10, early_stopping_min_delta: float = 1e-5, reduce_lr_patience: int = 4, reduce_lr_factor: float = 0.5, min_learning_rate: float = 1e-6, checkpoint_path: str | Path | None = "models/lstm_best.keras") -> None:
        units = [128, 64] if units is None else list(units)
        dropout_rates = [0.2, 0.2] if dropout_rates is None else list(dropout_rates)
        self._validate_config(units=units, dropout_rates=dropout_rates, dense_units=dense_units, learning_rate=learning_rate, clipnorm=clipnorm, epochs=epochs, batch_size=batch_size, validation_split=validation_split, early_stopping_patience=early_stopping_patience, early_stopping_min_delta=early_stopping_min_delta, reduce_lr_patience= reduce_lr_patience,reduce_lr_factor=reduce_lr_factor, min_learning_rate=min_learning_rate, checkpoint_path=checkpoint_path)

        self._units = units
        self._dropout_rates = dropout_rates
        self._dense_units = dense_units
        self._learning_rate = learning_rate
        self._clipnorm = clipnorm
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_min_delta = early_stopping_min_delta
        self._reduce_lr_patience = reduce_lr_patience
        self._reduce_lr_factor = reduce_lr_factor
        self._min_learning_rate = min_learning_rate
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._model: KerasModel | None = None


    @staticmethod
    def _validate_config(units: list[int], dropout_rates: list[float], dense_units: int, learning_rate: float, clipnorm: float, epochs: int, batch_size: int, validation_split: float, early_stopping_patience: int, early_stopping_min_delta: float, reduce_lr_patience: int, reduce_lr_factor: float, min_learning_rate: float, checkpoint_path: str | Path | None) -> None:
        if not units:
            raise ValueError("units cannot be empty.")
        if len(units) != len(dropout_rates):
            raise ValueError("units and dropout_rates must contain the same number of values.")

        for units_per_layer in units:
            if isinstance(units_per_layer, bool) or not isinstance(units_per_layer, int):
                raise TypeError("Every value in units must be an integer.")
            if units_per_layer <= 0:
                raise ValueError("Every value in units must be a positive integer.")
        for dropout_rate in dropout_rates:
            if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)):
                raise TypeError("Every dropout rate must be numeric.")
            if not 0 <= dropout_rate < 1:
                raise ValueError("Every dropout rate must satisfy 0 <= rate < 1.")

        positive_integer_parameters = {"dense_units": dense_units, "epochs": epochs, "batch_size": batch_size, "early_stopping_patience": early_stopping_patience, "reduce_lr_patience": reduce_lr_patience}
        positive_numeric_parameters = {"learning_rate": learning_rate, "clipnorm": clipnorm, "min_learning_rate": min_learning_rate}

        for name, value in positive_integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in positive_numeric_parameters.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if isinstance(validation_split, bool) or not isinstance(validation_split, (int, float)):
            raise TypeError("validation_split must be numeric.")
        if isinstance(early_stopping_min_delta, bool) or not isinstance(early_stopping_min_delta, (int, float)):
            raise TypeError("early_stopping_min_delta must be numeric.")
        if isinstance(reduce_lr_factor, bool) or not isinstance(reduce_lr_factor, (int, float)):
            raise TypeError("reduce_lr_factor must be numeric.")

        if not 0 < validation_split < 1:
            raise ValueError("validation_split must satisfy 0 < validation_split < 1.")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        if not 0 < reduce_lr_factor < 1:
            raise ValueError("reduce_lr_factor must satisfy 0 < factor < 1.")
        if min_learning_rate >= learning_rate:
            raise ValueError("min_learning_rate must be smaller than learning_rate.")

        if checkpoint_path is not None and not isinstance(checkpoint_path, (str, Path)):
            raise TypeError("checkpoint_path must be a string, Path, or None.")
    
    @property
    def keras_model(self) -> KerasModel | None:
        return self._model

    def _validate_model_input_shape(self, X: NDArray[np.number], X_name: str = "X")-> None:
        if self._model is None:
            raise RuntimeError("The model must be built or loaded first")
        
        expeted_shape = tuple(self._model.input_shape[1:])
        recieved_shape = tuple(X.shape[1:])

        if expeted_shape!=recieved_shape:
            raise ValueError(f"{X_name} shape does not match the model input shape.\nExpected {expeted_shape}, received {recieved_shape}.")
    

    def build(self, input_shape: tuple[int, int]) -> None:
        self._validate_input_shape(input_shape=input_shape)

        logger.info(f"Building LSTM model with input_shape={input_shape}, units={self._units}\ndropout_rates={self._dropout_rates}, dense_units={self._dense_units}")

        model = Sequential(name="LSTMStockPredictor")
        model.add(Input(shape=input_shape))

        for layer_index, (units, dropout_rate) in enumerate(zip(self._units, self._dropout_rates)):
            is_last_layer = layer_index == len(self._units) - 1

            model.add(LSTM(units=units, return_sequences= not is_last_layer, name = f"lstm_{layer_index + 1}"))

            if dropout_rate>0:
                model.add(Dropout(rate= dropout_rate, name= f"dropout_{layer_index + 1}"))
            
        model.add(Dense(units=self._dense_units, activation="relu", name= "dense_hidden"))
        model.add(Dense(units=1, activation="linear", name= "output"))

        optimizer = Adam(learning_rate=self._learning_rate, clipnorm=self._clipnorm)
        
        model.compile(optimizer=optimizer, loss=Huber(), metrics=[MeanAbsoluteError(name="mae")])

        self._model = model

        logger.info(f"LSTM model built successfully: input_shape={input_shape}, parameters={model.count_params():,}")

    def _create_callback(self)-> list[Any]:
        callback: list[Any] = [EarlyStopping(monitor="val_loss", mode="min", patience=self._early_stopping_patience, min_delta=self._early_stopping_min_delta, restore_best_weights= True, verbose=1), ReduceLROnPlateau(monitor="val_loss", mode="min", factor= self._reduce_lr_factor, patience= self._reduce_lr_patience, min_lr= self._min_learning_rate, verbose=1)]

        if self._checkpoint_path is not None:
            self._checkpoint_path.parent.mkdir(parents= True, exist_ok= True)
            callback.append(ModelCheckpoint(filepath=self._checkpoint_path, monitor= "val_loss", mode= "min", save_best_only=True, verbose=0))
        
        return callback
    
    def train(self, X: NDArray[np.number], y: NDArray[np.number])-> Any:
        if self._model is None:
            raise RuntimeError("Model must be Built or loaded before training")
        
        self._validate_training_data(X, y, X_name="X_train", y_name="y_train")
        self._validate_model_input_shape(X, X_name="X_train")
        
        validation_sample = int(X.shape[0] * self._validation_split)

        if validation_sample < 1:
            raise ValueError(f"validation_split={self._validation_split} produces no validation samples for {X.shape[0]} training samples.")
        if X.shape[0] - validation_sample < 1:
            raise ValueError(f"validation_split={self._validation_split} leaves no samples for training.")

        logger.info(f"Training LSTM model with samples={X.shape[0]}, epochs={self._epochs}, batch_size={self._batch_size}, validation_split={self._validation_split}")
        
        history = self._model.fit(X, y, epochs=self._epochs, batch_size= self._batch_size, validation_split= self._validation_split, shuffle=False, callbacks= self._create_callback(), verbose=1)

        best_validation_loss = min(history.history["val_loss"])
        best_validation_mae = min(history.history["val_mae"])

        logger.info(f"LSTM model training completed successfully: best_val_loss={best_validation_loss:.6f}, best_val_mae={best_validation_mae:.6f}")

        return history
    
    def predict(self, X: NDArray[np.number])-> NDArray[np.number]:
        if self._model is None:
            raise RuntimeError("The model must be build or loaded before Prediction Ayush")
        
        self._validate_prediction_data(X)
        self._validate_model_input_shape(X)
        
        logger.info(f"Generating predictions for {X.shape[0]} samples")

        prediction = self._model.predict(X, verbose=0)

        logger.info(f"Predictions generated successfully: shape={prediction.shape}")

        return prediction
    
    def save(self, path: str | Path)-> None:
        if self._model is None:
            raise RuntimeError("Ayush mode must be created or loaded before saving")
        
        if not isinstance(path, (str, Path)):
            raise TypeError("Path must be a string or a path")
        
        path = Path(path)
        path.parent.mkdir(parents= True, exist_ok= True)

        logger.info(f"Saving LSTM model to: {path}")

        self._model.save(path)

        logger.info(f"LSTM model saved successfully: {path}")

    def load(self, path: str | Path)-> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("Path must be a string or a path")
        
        path = Path(path)
        
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading LSTM model from: {path}")

        model = Keras_load_model(path)

        if len(model.input_shape) != 3:
            raise ValueError(f"The loaded model must accept 3D sequence input. Received input_shape={model.input_shape}.")
        if model.output_shape[-1] != 1:
            raise ValueError(f"The loaded model must produce exactly one output value per sample. Received output_shape={model.output_shape}.")
        
        self._model = model

        logger.info(f"LSTM model loaded successfully: input_shape={tuple(model.input_shape[1:])}")

class Model:
    def __init__(self, strategy: BaseModel)-> None:
        if not isinstance(strategy, BaseModel):
            raise TypeError(f"Expected BaseModel, got {type(strategy)}")
        logger.info(f"Setting the strategy for Model: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    @property
    def strategy(self) -> BaseModel:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseModel) -> None:
        if not isinstance(strategy, BaseModel):
            raise TypeError(f"Expected a BaseModel, got {type(strategy)}")
        logger.info(f"Model Creation — strategy set to: {type(strategy).__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: BaseModel)-> None:
        if not isinstance(strategy, BaseModel):
            raise TypeError(f"Expected BaseModel, got {type(strategy)}")

        logger.info(f"Changing the strategy for Model Creation: {strategy.__class__.__name__}")
        self._strategy = strategy
    
    def build(self, input_shape: tuple[int, int]) -> None:
        logger.info(f"Building model using strategy: {self._strategy.__class__.__name__}")
        self._strategy.build(input_shape)
    
    def train(self, X: NDArray[np.number], y: NDArray[np.number])-> Any:
        logger.info(f"Training model using strategy: {self._strategy.__class__.__name__}")
        return self._strategy.train(X,y)

    def predict(self, X: NDArray[np.number])->  NDArray[np.number]:
        logger.info(f"Generating predictions using strategy: {self._strategy.__class__.__name__}")
        return self._strategy.predict(X)
    
    def save(self, path: str | Path)-> None:
        logger.info(f"Saving model using strategy: {self._strategy.__class__.__name__}")
        self._strategy.save(path)

    def load(self, path: str | Path)-> None:
        logger.info(f"Loading model using strategy: {self._strategy.__class__.__name__}")
        self._strategy.load(path)