"""Evaluasi model prediksi kontrak bridge — 7 indikator C23 paper.

C23 paper (Lin et al., 2023) mendefinisikan 7 indikator:
  MS  — Most Suitable: prediksi tepat (suit + kategori sama)
  SCA — Same Category Acceptable: kategori sama, suit beda tapi masuk akal
  SCU — Same Category Unacceptable: kategori sama, suit tidak ideal
  SSE — Same Suit Excl. MS: suit sama, kategori beda
  O   — Others: tidak cocok sama sekali
  SC  — Same Category = MS + SCA + SCU  ← METRIK UTAMA (target ≥ 0.773)
  SS  — Same Suit = MS + SSE

IMP gain: selisih skor kontrak prediksi vs kontrak BBO, dikonversi ke IMP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    confusion_matrix, f1_score, accuracy_score,
)

FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")

# Tabel IMP resmi WBF
IMP_SCALE = [
    (20, 1), (50, 2), (90, 3), (130, 4), (170, 5),
    (220, 6), (270, 7), (320, 8), (370, 9), (430, 10),
    (500, 11), (600, 12), (750, 13), (900, 14), (1100, 15),
    (1300, 16), (1500, 17), (1750, 18), (2000, 19),
    (2250, 20), (2500, 21), (3000, 22), (3500, 23), (float("inf"), 24),
]

TRICK_VALUE = {"C": 20, "D": 20, "H": 30, "S": 30, "N": 30}


# ---------------------------------------------------------------------------
# Bridge scoring (untuk IMP)
# ---------------------------------------------------------------------------

def bridge_score(level: int, strain: str, tricks: int, vul: bool = False) -> int:
    """Hitung skor bridge (tanpa double/redouble)."""
    needed = level + 6
    if tricks < needed:
        return (tricks - needed) * (100 if vul else 50)

    trick_pts = (40 + 30 * (level - 1)) if strain == "N" else TRICK_VALUE[strain] * level
    overtricks = tricks - needed

    if level == 7:
        bonus = 1500 if vul else 1000
    elif level == 6:
        bonus = 750 if vul else 500
    elif trick_pts >= 100:
        bonus = 500 if vul else 300
    else:
        bonus = 50

    ot_val = 30 if strain == "N" else TRICK_VALUE[strain]
    return trick_pts + bonus + ot_val * overtricks


def point_diff_to_imp(diff: int) -> int:
    """Konversi selisih poin ke IMP menggunakan tabel WBF."""
    for threshold, imp in IMP_SCALE:
        if abs(diff) <= threshold:
            return imp
    return 24


# ---------------------------------------------------------------------------
# 7 Indikator C23
# ---------------------------------------------------------------------------

def classify_indicators(
    y_suit_true: pd.Series,
    y_cat_true: pd.Series,
    y_suit_pred: np.ndarray,
    y_cat_pred: np.ndarray,
) -> pd.DataFrame:
    """Klasifikasikan setiap prediksi ke salah satu dari 7 indikator C23.

    Catatan: SCA vs SCU memerlukan data 100-shuffle DDS untuk dibedakan secara
    penuh. Di sini, SC bukan MS diklasifikasikan sebagai SCA (pendekatan konservatif).
    """
    suit_match = np.array(y_suit_true) == y_suit_pred
    cat_match  = np.array(y_cat_true) == y_cat_pred

    indicators = []
    for sm, cm in zip(suit_match, cat_match):
        if sm and cm:
            indicators.append("MS")
        elif cm:
            indicators.append("SCA")   # Same Category, suit beda
        elif sm:
            indicators.append("SSE")   # Same Suit, kategori beda
        else:
            indicators.append("O")     # Sama sekali tidak cocok

    return pd.DataFrame({"indicator": indicators}, index=y_suit_true.index)


def indicator_summary(indicators: pd.DataFrame) -> pd.Series:
    """Hitung proporsi 7 indikator dan SC/SS gabungan.

    Returns:
        Series dengan keys: MS, SCA, SCU, SSE, O, SC, SS
    """
    counts = indicators["indicator"].value_counts()
    total = len(indicators)

    ms  = counts.get("MS",  0) / total
    sca = counts.get("SCA", 0) / total
    scu = counts.get("SCU", 0) / total
    sse = counts.get("SSE", 0) / total
    o   = counts.get("O",   0) / total

    sc = ms + sca + scu   # Same Category (metrik utama)
    ss = ms + sse          # Same Suit

    return pd.Series({
        "MS":           round(ms,  4),
        "SCA":          round(sca, 4),
        "SCU":          round(scu, 4),
        "SSE":          round(sse, 4),
        "O":            round(o,   4),
        "SC (utama)":   round(sc,  4),
        "SS":           round(ss,  4),
    })


# ---------------------------------------------------------------------------
# Evaluasi lengkap
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_test: pd.DataFrame,
    y_suit_test: pd.Series,
    y_cat_test: pd.Series,
    model_name: str = "TwoStageRF",
    save_figures: bool = True,
) -> dict:
    """Evaluasi lengkap: 7 indikator, F1, confusion matrix.

    Args:
        model: Model TwoStageRF yang sudah di-fit
        X_test: Feature matrix test set
        y_suit_test: Label suit yang benar
        y_cat_test: Label kategori yang benar
        model_name: Label untuk file yang disimpan
        save_figures: Simpan confusion matrix dan feature importance

    Returns:
        Dict berisi semua metrik evaluasi
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    y_suit_pred = model.predict_suit(X_test)
    y_cat_pred  = model.predict_category(X_test)

    # --- 7 indikator ---
    indicators = classify_indicators(y_suit_test, y_cat_test, y_suit_pred, y_cat_pred)
    summary    = indicator_summary(indicators)

    print("\n" + "="*50)
    print("  7-INDIKATOR EVALUASI (C23 Tabel 4)")
    print("="*50)
    for key, val in summary.items():
        bar = "█" * int(val * 30)
        marker = " ← METRIK UTAMA" if "SC" in key else ""
        print(f"  {key:<15} {val:.4f}  {bar}{marker}")
    print(f"\n  Target SC paper: 0.773 | Hasil: {summary['SC (utama)']:.4f}")

    # --- Per-stage metrics ---
    suit_f1 = f1_score(y_suit_test, y_suit_pred, average="weighted", zero_division=0)
    cat_f1  = f1_score(y_cat_test,  y_cat_pred,  average="weighted", zero_division=0)
    suit_acc = accuracy_score(y_suit_test, y_suit_pred)
    cat_acc  = accuracy_score(y_cat_test,  y_cat_pred)

    print(f"\n  Stage 1 (Suit)     F1={suit_f1:.4f}  Acc={suit_acc:.4f}")
    print(f"  Stage 2 (Category) F1={cat_f1:.4f}  Acc={cat_acc:.4f}")

    # --- Simpan classification report ---
    report_suit = classification_report(y_suit_test, y_suit_pred, zero_division=0)
    report_cat  = classification_report(y_cat_test,  y_cat_pred,  zero_division=0)
    report_text = (
        f"=== {model_name} — Stage 1: Suit ===\n{report_suit}\n"
        f"=== {model_name} — Stage 2: Category ===\n{report_cat}\n"
        f"=== 7-Indikator Summary ===\n{summary.to_string()}\n"
    )
    (METRICS_DIR / f"classification_report_{model_name}.txt").write_text(
        report_text, encoding="utf-8"
    )

    if save_figures:
        # Confusion matrices
        plot_confusion_matrix(y_suit_test, y_suit_pred, f"{model_name}_suit")
        plot_confusion_matrix(y_cat_test,  y_cat_pred,  f"{model_name}_category")

        # Feature importance (jika model menyimpan feature_names_)
        if hasattr(model, "feature_importance"):
            imp_suit, imp_cat = model.feature_importance()
            plot_feature_importance(imp_suit, imp_cat, top_n=20, model_name=model_name)

    return {
        "sc_accuracy":    summary["SC (utama)"],
        "ss_accuracy":    summary["SS"],
        "ms_accuracy":    summary["MS"],
        "suit_f1":        suit_f1,
        "category_f1":    cat_f1,
        "suit_accuracy":  suit_acc,
        "cat_accuracy":   cat_acc,
        "indicators":     summary,
    }


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    normalize: bool = False,
) -> None:
    """Plot dan simpan confusion matrix."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(list(y_true.unique()) + list(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, colorbar=True, values_format=".2f" if normalize else "d"
    )
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    out = FIGURES_DIR / f"confusion_matrix_{title.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix disimpan: {out}")


def plot_feature_importance(
    imp_suit: pd.Series,
    imp_cat: pd.Series,
    top_n: int = 20,
    model_name: str = "TwoStageRF",
) -> None:
    """Plot feature importance kedua stage (C23 Figure 4 style)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, imp, title in [
        (axes[0], imp_suit, f"Stage 1: Top-{top_n} Suit Predictor"),
        (axes[1], imp_cat,  f"Stage 2: Top-{top_n} Category Predictor"),
    ]:
        imp.head(top_n)[::-1].plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Feature Importance")

    plt.suptitle(f"Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Feature importance disimpan: {out}")


def plot_indicator_bar(summary: pd.Series, model_name: str = "TwoStageRF") -> None:
    """Bar chart proporsi 7 indikator."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#3498db", "#9b59b6"]
    keys = ["MS", "SCA", "SCU", "SSE", "O", "SC (utama)", "SS"]
    vals = [summary.get(k, 0) for k in keys]

    bars = ax.bar(keys, vals, color=colors[:len(keys)])
    ax.axhline(0.773, color="red", linestyle="--", linewidth=1, label="Paper target SC=0.773")
    ax.set_ylabel("Proporsi")
    ax.set_title(f"7-Indikator Evaluasi — {model_name}")
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / f"indicators_{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Indicator chart disimpan: {out}")
