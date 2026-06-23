"""2-Stage Neural Network (MLP + LSTM) untuk prediksi kontrak bridge terbaik.

Arsitektur:
  Stage 1 — MLP/LSTM prediksi suit (C/D/H/S/N)
  Stage 2 — MLP/LSTM prediksi kategori (partscore/game/small_slam/grand_slam)
  Kontrak final = kombinasi suit + kategori → level minimum yang valid

Model tersedia:
  - TwoStageMLP: Multi-Layer Perceptron (lebih cocok untuk tabular features)
  - TwoStageLSTM: LSTM (untuk sequence/temporal features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Level minimum untuk mencapai game per suit
GAME_LEVEL = {"C": 5, "D": 5, "H": 4, "S": 4, "N": 3}

# Hyperparameter default
NN_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "verbose": 0,
}

MODEL_PATH = Path("results/metrics/nn_model.h5")


# ---------------------------------------------------------------------------
# Model Architecture — MLP
# ---------------------------------------------------------------------------

class TwoStageMLP:
    """2-Stage MLP untuk prediksi kontrak bridge.
    
    Stage 1: MLP → prediksi suit (C/D/H/S/N)
    Stage 2: MLP → prediksi kategori (partscore/game/small_slam/grand_slam)
    
    Cocok untuk tabular features.
    """

    def __init__(self, input_dim: int, hidden_units: list = None, params: dict = None) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [256, 128, 64]
        self.params = {**NN_PARAMS, **(params or {})}
        
        self.model_suit = None
        self.model_category = None
        self.scaler = None
        self.encoder_suit = None
        self.encoder_category = None
        self.feature_names_ = None
        self.suit_classes_ = None
        self.category_classes_ = None

    def _build_mlp(self, output_classes: int, model_name: str = "mlp") -> keras.Model:
        """Build MLP architecture."""
        model = keras.Sequential(name=model_name)
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(0.3))
        
        # Output layer
        if output_classes == 2:
            model.add(layers.Dense(output_classes, activation="sigmoid"))
            loss_fn = "binary_crossentropy"
        else:
            model.add(layers.Dense(output_classes, activation="softmax"))
            loss_fn = "categorical_crossentropy"
        
        model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=["accuracy"]
        )
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "TwoStageMLP":
        """Latih kedua stage secara independen."""
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.encoder_suit = LabelEncoder()
        self.encoder_category = LabelEncoder()
        
        y_suit_encoded = self.encoder_suit.fit_transform(y_suit)
        y_category_encoded = self.encoder_category.fit_transform(y_category)
        
        self.suit_classes_ = self.encoder_suit.classes_
        self.category_classes_ = self.encoder_category.classes_
        
        # Convert to one-hot encoding
        n_suit_classes = len(self.suit_classes_)
        n_cat_classes = len(self.category_classes_)
        
        y_suit_onehot = keras.utils.to_categorical(y_suit_encoded, n_suit_classes)
        y_cat_onehot = keras.utils.to_categorical(y_category_encoded, n_cat_classes)
        
        # Build and train models
        print("Building Stage 1 (suit predictor) - MLP...")
        self.model_suit = self._build_mlp(n_suit_classes, "suit_mlp")
        
        print("Training Stage 1 (suit)...")
        self.model_suit.fit(
            X_scaled, y_suit_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        print("Building Stage 2 (category predictor) - MLP...")
        self.model_category = self._build_mlp(n_cat_classes, "category_mlp")
        
        print("Training Stage 2 (category)...")
        self.model_category.fit(
            X_scaled, y_cat_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediksi suit, kategori, dan level kontrak final."""
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        y_suit_probs = self.model_suit.predict(X_scaled, verbose=0)
        y_cat_probs = self.model_category.predict(X_scaled, verbose=0)
        
        # Get class predictions
        y_suit_pred_idx = np.argmax(y_suit_probs, axis=1)
        y_cat_pred_idx = np.argmax(y_cat_probs, axis=1)
        
        pred_suit = self.encoder_suit.inverse_transform(y_suit_pred_idx)
        pred_category = self.encoder_category.inverse_transform(y_cat_pred_idx)
        
        pred_level = _category_to_level(pred_suit, pred_category)
        pred_contract = [
            f"{lvl}{suit}" if suit != "P" else "PASS"
            for lvl, suit in zip(pred_level, pred_suit)
        ]
        
        return pd.DataFrame({
            "pred_suit":     pred_suit,
            "pred_category": pred_category,
            "pred_level":    pred_level,
            "pred_contract": pred_contract,
        }, index=X.index if hasattr(X, "index") else None)

    def predict_suit(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = np.argmax(self.model_suit.predict(X_scaled, verbose=0), axis=1)
        return self.encoder_suit.inverse_transform(y_pred_idx)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = np.argmax(self.model_category.predict(X_scaled, verbose=0), axis=1)
        return self.encoder_category.inverse_transform(y_pred_idx)


# ---------------------------------------------------------------------------
# Model Architecture — LSTM
# ---------------------------------------------------------------------------

class TwoStageLSTM:
    """2-Stage LSTM untuk prediksi kontrak bridge.
    
    Stage 1: LSTM → prediksi suit
    Stage 2: LSTM → prediksi kategori
    
    Features direshape sebagai sequence untuk LSTM.
    """

    def __init__(self, input_dim: int, lstm_units: list = None, params: dict = None) -> None:
        self.input_dim = input_dim
        self.lstm_units = lstm_units or [128, 64]
        self.params = {**NN_PARAMS, **(params or {})}
        self.seq_length = 10  # Reshape features menjadi sequence
        
        self.model_suit = None
        self.model_category = None
        self.scaler = None
        self.encoder_suit = None
        self.encoder_category = None
        self.feature_names_ = None
        self.suit_classes_ = None
        self.category_classes_ = None

    def _reshape_to_sequence(self, X: np.ndarray) -> np.ndarray:
        """Reshape tabular features menjadi sequence."""
        n_samples = X.shape[0]
        # Pad atau truncate features to seq_length
        n_features_per_step = max(1, self.input_dim // self.seq_length)
        reshaped = np.zeros((n_samples, self.seq_length, n_features_per_step))
        
        for i in range(min(self.seq_length * n_features_per_step, self.input_dim)):
            seq_idx = i // n_features_per_step
            feat_idx = i % n_features_per_step
            reshaped[:, seq_idx, feat_idx] = X[:, i]
        
        return reshaped

    def _build_lstm(self, output_classes: int, model_name: str = "lstm") -> keras.Model:
        """Build LSTM architecture."""
        n_features_per_step = max(1, self.input_dim // self.seq_length)
        
        model = keras.Sequential(name=model_name)
        model.add(layers.Input(shape=(self.seq_length, n_features_per_step)))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(0.3))
        
        # Dense layers
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))
        
        # Output layer
        if output_classes == 2:
            model.add(layers.Dense(output_classes, activation="sigmoid"))
            loss_fn = "binary_crossentropy"
        else:
            model.add(layers.Dense(output_classes, activation="softmax"))
            loss_fn = "categorical_crossentropy"
        
        model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=["accuracy"]
        )
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "TwoStageLSTM":
        """Latih kedua stage LSTM secara independen."""
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape to sequence
        X_seq = self._reshape_to_sequence(X_scaled)
        
        # Encode labels
        self.encoder_suit = LabelEncoder()
        self.encoder_category = LabelEncoder()
        
        y_suit_encoded = self.encoder_suit.fit_transform(y_suit)
        y_category_encoded = self.encoder_category.fit_transform(y_category)
        
        self.suit_classes_ = self.encoder_suit.classes_
        self.category_classes_ = self.encoder_category.classes_
        
        # Convert to one-hot encoding
        n_suit_classes = len(self.suit_classes_)
        n_cat_classes = len(self.category_classes_)
        
        y_suit_onehot = keras.utils.to_categorical(y_suit_encoded, n_suit_classes)
        y_cat_onehot = keras.utils.to_categorical(y_category_encoded, n_cat_classes)
        
        # Build and train models
        print("Building Stage 1 (suit predictor) - LSTM...")
        self.model_suit = self._build_lstm(n_suit_classes, "suit_lstm")
        
        print("Training Stage 1 (suit)...")
        self.model_suit.fit(
            X_seq, y_suit_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        print("Building Stage 2 (category predictor) - LSTM...")
        self.model_category = self._build_lstm(n_cat_classes, "category_lstm")
        
        print("Training Stage 2 (category)...")
        self.model_category.fit(
            X_seq, y_cat_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediksi suit, kategori, dan level kontrak final."""
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        
        # Predict probabilities
        y_suit_probs = self.model_suit.predict(X_seq, verbose=0)
        y_cat_probs = self.model_category.predict(X_seq, verbose=0)
        
        # Get class predictions
        y_suit_pred_idx = np.argmax(y_suit_probs, axis=1)
        y_cat_pred_idx = np.argmax(y_cat_probs, axis=1)
        
        pred_suit = self.encoder_suit.inverse_transform(y_suit_pred_idx)
        pred_category = self.encoder_category.inverse_transform(y_cat_pred_idx)
        
        pred_level = _category_to_level(pred_suit, pred_category)
        pred_contract = [
            f"{lvl}{suit}" if suit != "P" else "PASS"
            for lvl, suit in zip(pred_level, pred_suit)
        ]
        
        return pd.DataFrame({
            "pred_suit":     pred_suit,
            "pred_category": pred_category,
            "pred_level":    pred_level,
            "pred_contract": pred_contract,
        }, index=X.index if hasattr(X, "index") else None)

    def predict_suit(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        y_pred_idx = np.argmax(self.model_suit.predict(X_seq, verbose=0), axis=1)
        return self.encoder_suit.inverse_transform(y_pred_idx)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        y_pred_idx = np.argmax(self.model_category.predict(X_seq, verbose=0), axis=1)
        return self.encoder_category.inverse_transform(y_pred_idx)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _category_to_level(suits: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Konversi (suit, category) ke level kontrak minimum yang valid."""
    levels = []
    for suit, cat in zip(suits, categories):
        if cat == "grand_slam":
            levels.append(7)
        elif cat == "small_slam":
            levels.append(6)
        elif cat == "game":
            levels.append(GAME_LEVEL.get(suit, 3))
        else:  # partscore
            levels.append(max(1, GAME_LEVEL.get(suit, 3) - 1))
    return np.array(levels)


def train(
    X_train: pd.DataFrame,
    y_suit_train: pd.Series,
    y_category_train: pd.Series,
    model_type: str = "mlp",
    params: dict = None,
) -> Union[TwoStageMLP, TwoStageLSTM]:
    """Latih model neural network dengan tipe yang dipilih.
    
    Args:
        X_train: Features train
        y_suit_train: Target suit train
        y_category_train: Target category train
        model_type: "mlp" atau "lstm"
        params: Hyperparameter override
    
    Returns:
        Model yang sudah dilatih
    """
    params = params or NN_PARAMS
    
    if model_type.lower() == "mlp":
        print("=" * 60)
        print("Training 2-Stage MLP Model")
        print("=" * 60)
        model = TwoStageMLP(input_dim=X_train.shape[1], params=params)
    elif model_type.lower() == "lstm":
        print("=" * 60)
        print("Training 2-Stage LSTM Model")
        print("=" * 60)
        model = TwoStageLSTM(input_dim=X_train.shape[1], params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'lstm'.")
    
    model.fit(X_train, y_suit_train, y_category_train)
    print("Training selesai.\n")
    return model


def save_model(model: Union[TwoStageMLP, TwoStageLSTM], path: Path = MODEL_PATH) -> None:
    """Simpan model ke disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if isinstance(model, TwoStageMLP):
        model.model_suit.save(path.parent / "model_suit.h5")
        model.model_category.save(path.parent / "model_category.h5")
    elif isinstance(model, TwoStageLSTM):
        model.model_suit.save(path.parent / "model_suit_lstm.h5")
        model.model_category.save(path.parent / "model_category_lstm.h5")
    
    # Save encoders and scaler
    joblib.dump(model.scaler, path.parent / "scaler.pkl")
    joblib.dump(model.encoder_suit, path.parent / "encoder_suit.pkl")
    joblib.dump(model.encoder_category, path.parent / "encoder_category.pkl")
    
    print(f"Model disimpan ke {path.parent}")


def load_model(model_type: str = "mlp", path: Path = MODEL_PATH) -> Union[TwoStageMLP, TwoStageLSTM]:
    """Load model dari disk."""
    path_dir = path.parent
    
    if model_type.lower() == "mlp":
        model = TwoStageMLP(input_dim=98)  # Dummy, akan di-load
        model.model_suit = keras.models.load_model(path_dir / "model_suit.h5")
        model.model_category = keras.models.load_model(path_dir / "model_category.h5")
    elif model_type.lower() == "lstm":
        model = TwoStageLSTM(input_dim=98)  # Dummy, akan di-load
        model.model_suit = keras.models.load_model(path_dir / "model_suit_lstm.h5")
        model.model_category = keras.models.load_model(path_dir / "model_category_lstm.h5")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'lstm'.")
    
    model.scaler = joblib.load(path_dir / "scaler.pkl")
    model.encoder_suit = joblib.load(path_dir / "encoder_suit.pkl")
    model.encoder_category = joblib.load(path_dir / "encoder_category.pkl")
    
    return model


# ---------------------------------------------------------------------------
# Persiapan feature matrix
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Pilih kolom fitur dari dataset, isi missing dengan 0."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    return X


if __name__ == "__main__":
    from src.features import FEATURE_COLS
    from sklearn.model_selection import train_test_split

    processed_csv = Path("data/processed/bridge_dataset.csv")
    df = pd.read_csv(processed_csv)
    df = df.dropna(subset=["best_contract_strain", "best_contract_category"])
    df = df[df["best_contract_strain"] != "P"]

    X = prepare_features(df, FEATURE_COLS)
    y_suit = df["best_contract_strain"]
    y_cat  = df["best_contract_category"]

    print(f"Dataset: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"Distribusi suit:\n{y_suit.value_counts()}")
    print(f"Distribusi kategori:\n{y_cat.value_counts()}")

    X_train, X_test, y_suit_train, y_suit_test = train_test_split(X, y_suit, test_size=0.2, stratify=y_suit, random_state=42)
    _, _, y_cat_train, y_cat_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_suit, random_state=42)

    # Train MLP
    model = train(X_train, y_suit_train, y_cat_train, model_type="mlp")
    save_model(model)
