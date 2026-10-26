"""
Step 12c - Results Pipeline: figures and metrics tables for the paper.

Retrains the same baseline models as training.py (same data, same split - see
load_data()/make_group_split() there) and produces the standard set of
research-paper figures for both tasks, plus a metrics summary table.

Uses scikit-learn's built-in Display classes (ConfusionMatrixDisplay,
RocCurveDisplay, PrecisionRecallDisplay) rather than hand-rolled plotting code -
these are the standard, reviewer-recognizable way to present these curves.

Output (all in reports/figures/ and reports/):
  regression_actual_vs_predicted.png   - 3-panel scatter, one per model
  regression_residuals.png             - 3-panel residual plot
  regression_feature_importance.png    - RF + XGB feature importances
  regression_metrics_comparison.png    - MAE / RMSE / R2 bar charts
  classification_confusion_matrices.png
  classification_roc_curves.png        - all 3 models overlaid
  classification_pr_curves.png         - all 3 models overlaid
  classification_feature_importance.png
  classification_metrics_comparison.png
  metrics_summary.csv / metrics_summary.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

from training import load_data, make_group_split, train_decision_quality_models, train_blunder_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Decision Quality (regression) figures
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        preds = r["preds"]
        ax.scatter(y_test, preds, alpha=0.3, s=10)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)  # perfect-prediction reference line
        ax.set_title(name)
        ax.set_xlabel("Actual decision quality")
    axes[0].set_ylabel("Predicted decision quality")
    fig.suptitle("Decision Quality: Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residuals(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        residuals = y_test.values - r["preds"]
        ax.scatter(r["preds"], residuals, alpha=0.3, s=10)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Predicted decision quality")
    axes[0].set_ylabel("Residual (actual - predicted)")
    fig.suptitle("Decision Quality: Residuals")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_residuals.png", dpi=150)
    plt.close(fig)


def plot_regression_metrics(results):
    names = list(results.keys())
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        values = [results[n][metric] for n in names]
        ax.bar(names, values)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Decision Quality: Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_metrics_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Blunder Probability (classification) figures
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    for ax, (name, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["preds"], ax=ax, colorbar=False, display_labels=["No Blunder", "Blunder"]
        )
        ax.set_title(name)
    fig.suptitle("Blunder Probability: Confusion Matrices")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        RocCurveDisplay.from_predictions(y_test, r["probs"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
    ax.set_title("Blunder Probability: ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        PrecisionRecallDisplay.from_predictions(y_test, r["probs"], name=name, ax=ax)
    base_rate = y_test.mean()
    ax.axhline(base_rate, color="k", linestyle="--", linewidth=1, label=f"Base rate ({base_rate:.2f})")
    ax.set_title("Blunder Probability: Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_classification_metrics(results):
    names = list(results.keys())
    metrics = ["precision", "recall", "f1", "auc"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, metric in zip(axes, metrics):
        values = [results[n][metric] for n in names]
        ax.bar(names, values)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Blunder Probability: Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_metrics_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature importance (tree-based models only - Linear/Logistic coefficients
# aren't on the same scale as tree importances, so they're left out here
# rather than plotted misleadingly next to them)
# ---------------------------------------------------------------------------

def plot_feature_importance(results, feature_names, title, save_name):
    tree_models = {n: r for n, r in results.items() if hasattr(r["model"], "feature_importances_")}
    if not tree_models:
        return
    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, tree_models.items()):
        importances = pd.Series(r["model"].feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=True).tail(15)  # top 15, easiest to read
        ax.barh(importances.index, importances.values)
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def save_metrics_table(quality_results, blunder_results):
    quality_rows = [
        {"model": name, "task": "decision_quality", **{k: v for k, v in r.items() if k not in ("model", "preds", "probs")}}
        for name, r in quality_results.items()
    ]
    blunder_rows = [
        {"model": name, "task": "blunder", **{k: v for k, v in r.items() if k not in ("model", "preds", "probs")}}
        for name, r in blunder_results.items()
    ]
    df = pd.DataFrame(quality_rows + blunder_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)

    # Written by hand instead of df.to_markdown() - that method needs the
    # optional "tabulate" package, which isn't in requirements.txt.
    cols = df.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    (REPORTS_DIR / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(df)
    return df


# ---------------------------------------------------------------------------

def _attach_predictions(results, X_test, y_test, task):
    """train_*_models() in training.py doesn't keep predictions around - refit
    already happened there, so recompute predict()/predict_proba() once here
    rather than duplicating the training loop."""
    for name, r in results.items():
        r["preds"] = r["model"].predict(X_test)
        if task == "classification":
            r["probs"] = r["model"].predict_proba(X_test)[:, 1]
    return results


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]

    print("Training decision quality models...")
    quality_results = train_decision_quality_models(X_train, X_test, y1_train, y1_test)
    quality_results = _attach_predictions(quality_results, X_test, y1_test, "regression")

    print("Training blunder models...")
    blunder_results = train_blunder_models(X_train, X_test, y2_train, y2_test)
    blunder_results = _attach_predictions(blunder_results, X_test, y2_test, "classification")

    print("Generating regression figures...")
    plot_actual_vs_predicted(quality_results, y1_test)
    plot_residuals(quality_results, y1_test)
    plot_regression_metrics(quality_results)
    plot_feature_importance(quality_results, X.columns, "Decision Quality: Feature Importance",
                             "regression_feature_importance.png")

    print("Generating classification figures...")
    plot_confusion_matrices(blunder_results, y2_test)
    plot_roc_curves(blunder_results, y2_test)
    plot_pr_curves(blunder_results, y2_test)
    plot_classification_metrics(blunder_results)
    plot_feature_importance(blunder_results, X.columns, "Blunder Probability: Feature Importance",
                             "classification_feature_importance.png")

    print("Saving metrics table...")
    save_metrics_table(quality_results, blunder_results)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print(f"Metrics table saved to {REPORTS_DIR}/metrics_summary.csv and .md")


if __name__ == "__main__":
    main()
