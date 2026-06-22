"""2-Stage Random Forest untuk prediksi kontrak bridge terbaik.

Arsitektur mengikuti C23 paper (Lin et al., 2023):
  Stage 1 — RF prediksi suit (C/D/H/S/N)
  Stage 2 — RF prediksi kategori (partscore/game/small_slam/grand_slam)
  Kontrak final = kombinasi suit + kategori → level minimum yang valid

Evaluasi mengikuti paper:
  5-fold cross-validation diulang 10 kali (RepeatedStratifiedKFold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Level minimum untuk mencapai game per suit
GAME_LEVEL = {"C": 5, "D": 5, "H": 4, "S": 4, "N": 3}

# Hyperparameter default RF (sesuai rekomendasi C23)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_PATH = Path("results/metrics/rf_model.pkl")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoStageRF:
    """2-Stage Random Forest untuk prediksi kontrak bridge.

    Stage 1: RandomForest → prediksi suit (C/D/H/S/N)
    Stage 2: RandomForest → prediksi kategori (partscore/game/small_slam/grand_slam)
    """

    def __init__(self, params: dict = RF_PARAMS) -> None:
        self.rf_suit     = RandomForestClassifier(**params)
        self.rf_category = RandomForestClassifier(**params)
        self.feature_names_: Optional[list] = None

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "TwoStageRF":
        """Latih kedua stage secara independen."""
        self.feature_names_ = list(X.columns)
        self.rf_suit.fit(X, y_suit)
        self.rf_category.fit(X, y_category)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediksi suit, kategori, dan level kontrak final.

        Returns:
            DataFrame kolom: pred_suit, pred_category, pred_level, pred_contract
        """
        pred_suit     = self.rf_suit.predict(X)
        pred_category = self.rf_category.predict(X)
        pred_level    = _category_to_level(pred_suit, pred_category)
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
        return self.rf_suit.predict(X)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf_category.predict(X)

    def feature_importance(self) -> Tuple[pd.Series, pd.Series]:
        """Return feature importances untuk kedua stage (sorted descending)."""
        names = self.feature_names_ or []
        suit_imp = pd.Series(
            self.rf_suit.feature_importances_, index=names
        ).sort_values(ascending=False)
        cat_imp = pd.Series(
            self.rf_category.feature_importances_, index=names
        ).sort_values(ascending=False)
        return suit_imp, cat_imp


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


# ---------------------------------------------------------------------------
# Training & Cross-Validation
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset menjadi train dan test (stratified)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def cross_validate(
    X: pd.DataFrame,
    y_suit: pd.Series,
    y_category: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """5-fold CV × 10 repeats sesuai C23 paper Section 5.

    Returns:
        Dict dengan rata-rata dan std F1-weighted untuk suit dan category.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    rf_suit = RandomForestClassifier(**RF_PARAMS)
    rf_cat  = RandomForestClassifier(**RF_PARAMS)

    print(f"Cross-validating Stage 1 (suit): {n_splits}-fold × {n_repeats} repeats...")
    scores_suit = cross_val_score(rf_suit, X, y_suit, cv=cv, scoring="f1_weighted", n_jobs=-1)

    print(f"Cross-validating Stage 2 (category): {n_splits}-fold × {n_repeats} repeats...")
    scores_cat = cross_val_score(rf_cat, X, y_category, cv=cv, scoring="f1_weighted", n_jobs=-1)

    return {
        "suit_f1_mean":     round(scores_suit.mean(), 4),
        "suit_f1_std":      round(scores_suit.std(), 4),
        "category_f1_mean": round(scores_cat.mean(), 4),
        "category_f1_std":  round(scores_cat.std(), 4),
        "n_splits":  n_splits,
        "n_repeats": n_repeats,
    }


def train(
    X_train: pd.DataFrame,
    y_suit_train: pd.Series,
    y_category_train: pd.Series,
) -> TwoStageRF:
    """Latih model 2-stage RF dengan hyperparameter default."""
    print("Training Stage 1 (suit predictor)...")
    model = TwoStageRF()
    model.fit(X_train, y_suit_train, y_category_train)
    print("Training selesai.")
    return model


def save_model(model: TwoStageRF, path: Path = MODEL_PATH) -> None:
    """Simpan model ke disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model disimpan ke {path}")


def load_model(path: Path = MODEL_PATH) -> TwoStageRF:
    """Load model dari disk."""
    return joblib.load(path)


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

    processed_csv = Path("data/processed/bridge_dataset.csv")
    df = pd.read_csv(processed_csv)
    df = df.dropna(subset=["best_contract_strain", "best_contract_category"])
    df = df[df["best_contract_strain"] != "P"]  # keluarkan board pass-out

    X = prepare_features(df, FEATURE_COLS)
    y_suit = df["best_contract_strain"]
    y_cat  = df["best_contract_category"]

    print(f"Dataset: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"Distribusi suit:\n{y_suit.value_counts()}")
    print(f"Distribusi kategori:\n{y_cat.value_counts()}")

    cv_results = cross_validate(X, y_suit, y_cat)
    print(f"\nCV Suit F1:     {cv_results['suit_f1_mean']:.4f} ± {cv_results['suit_f1_std']:.4f}")
    print(f"CV Category F1: {cv_results['category_f1_mean']:.4f} ± {cv_results['category_f1_std']:.4f}")

    X_train, X_test, y_suit_train, y_suit_test = split_data(X, y_suit)
    _, _, y_cat_train, y_cat_test = split_data(X, y_cat)

    model = train(X_train, y_suit_train, y_cat_train)
    save_model(model)
