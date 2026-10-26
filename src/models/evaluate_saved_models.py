"""
Step 12e - Report-Card Evaluation of the SAVED models

Different from training.py (trains 3 candidates per target and picks the
best) and from the dashboard's model-comparison tabs (which retrain fresh
candidates every session for a fair side-by-side comparison): this script
evaluates ONLY the two models actually saved to models/*.pkl - the ones
that would actually ship. No retraining happens here.

It also does NOT trust a fixed 0.5 cutoff for the blunder model. Blunders
are a small minority class (~5% of moves), so this sweeps thresholds and
reports the F1-optimal one alongside the default - see docs/step12_model_training.md
for why the default cutoff is not meaningful here.

Evaluates purely on model performance against whatever is currently in
data/processed/ - no dataset-provenance checking here (see improve_pipeline.py
if you need the tuning/model-selection pipeline instead of just scoring).

Produces:
  - reports/report_card_summary.csv / report_card_summary.md   (paper-ready headline numbers,
  kept separate from reports/metrics_summary.csv/.md, which is evaluation.py's own
  baseline-comparison report - different purpose, different file)
  - reports/report_card_threshold_sweep.csv                        (full precision/recall/F1
                                                          sweep for the blunder model)
  - reports/figures/report_card_*.png                   (PR curve, ROC curve,
                                                          confusion matrices,
                                                          regression scatter)

Run:
    venv\\Scripts\\python.exe src\\models\\evaluate_saved_models.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training import load_data, make_group_split  # noqa: E402
from prediction import load_models  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def evaluate_regression(model, X_test, y_test, y_train) -> dict:
    preds = model.predict(X_test)
    naive_preds = np.full(len(y_test), y_train.mean())
    return {
        "model": type(model).__name__,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
        "naive_baseline_MAE": mean_absolute_error(y_test, naive_preds),
        "y_test_std": y_test.std(),
        "preds": preds,
    }


def sweep_thresholds(y_test, probs, n_steps: int = 199) -> pd.DataFrame:
    """Precision/recall/F1 at a grid of thresholds - the shape of this curve
    matters more than any single number for a rare-event classifier."""
    rows = []
    for t in np.linspace(0.005, 0.995, n_steps):
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "flagged_rate": preds.mean(),
        })
    return pd.DataFrame(rows)


def evaluate_classification(model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds_default = (probs >= 0.5).astype(int)

    sweep = sweep_thresholds(y_test, probs)
    best_row = sweep.loc[sweep["f1"].idxmax()]
    preds_tuned = (probs >= best_row["threshold"]).astype(int)

    return {
        "model": type(model).__name__,
        "auc": roc_auc_score(y_test, probs),
        "base_rate": y_test.mean(),
        "default": {
            "threshold": 0.5,
            "precision": precision_score(y_test, preds_default, zero_division=0),
            "recall": recall_score(y_test, preds_default, zero_division=0),
            "f1": f1_score(y_test, preds_default, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds_default),
        },
        "tuned": {
            "threshold": float(best_row["threshold"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "confusion_matrix": confusion_matrix(y_test, preds_tuned),
        },
        "sweep": sweep,
        "probs": probs,
    }


def make_figures(q_result, b_result, y_test_q, y_test_b):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Regression: actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_q, q_result["preds"], alpha=0.15, s=8)
    lims = [0, 100]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual decision quality")
    ax.set_ylabel("Predicted decision quality")
    ax.set_title(f"{q_result['model']}: Actual vs Predicted (R2={q_result['R2']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "report_card_regression_scatter.png", dpi=150)
    plt.close(fig)

    # Classification: PR curve + threshold sweep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    precision, recall, _ = precision_recall_curve(y_test_b, b_result["probs"])
    axes[0].plot(recall, precision, label=f"{b_result['model']} (AUC={b_result['auc']:.3f})")
    axes[0].axhline(b_result["base_rate"], color="gray", linestyle="--", label="base rate")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()

    sweep = b_result["sweep"]
    axes[1].plot(sweep["threshold"], sweep["precision"], label="precision")
    axes[1].plot(sweep["threshold"], sweep["recall"], label="recall")
    axes[1].plot(sweep["threshold"], sweep["f1"], label="F1", linewidth=2)
    axes[1].axvline(b_result["tuned"]["threshold"], color="red", linestyle="--",
                     label=f"best F1 @ {b_result['tuned']['threshold']:.2f}")
    axes[1].axvline(0.5, color="gray", linestyle=":", label="default 0.5")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Threshold Sweep")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "report_card_blunder_thresholds.png", dpi=150)
    plt.close(fig)

    # Confusion matrices, default vs tuned
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, title in zip(axes, ["default", "tuned"],
                               [f"Default (t=0.5)", f"Tuned (t={b_result['tuned']['threshold']:.2f})"]):
        cm = b_result[key]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No blunder", "Blunder"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No blunder", "Blunder"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{b_result['model']}: {title}\nF1={b_result[key]['f1']:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "report_card_confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading saved models (no retraining)...")
    quality_model, blunder_model, quality_path, blunder_path = load_models()
    print(f"  Decision quality: {quality_path.name} ({type(quality_model).__name__})")
    print(f"  Blunder:          {blunder_path.name} ({type(blunder_model).__name__})")

    print("\nLoading data/processed/ and rebuilding the same group split "
          "(random_state=42, test_size=0.2)...")
    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]

    n_games = game_id.nunique()
    print(f"  Data on disk: {len(X):,} moves / {n_games:,} games "
          f"(train {len(X_train):,} / test {len(X_test):,})")

    print("\n" + "=" * 70)
    print("DECISION QUALITY (regression)")
    print("=" * 70)
    q_result = evaluate_regression(quality_model, X_test, y1_test, y1_train)
    print(f"MAE  = {q_result['MAE']:.3f}   (naive baseline MAE = {q_result['naive_baseline_MAE']:.3f})")
    print(f"RMSE = {q_result['RMSE']:.3f}")
    print(f"R2   = {q_result['R2']:.4f}")
    print(f"test-set y std = {q_result['y_test_std']:.2f}  "
          "(R2 is the fair cross-run comparison metric, not MAE - MAE shrinks "
          "automatically on a lower-variance test slice even with no model improvement)")

    print("\n" + "=" * 70)
    print("BLUNDER (classification)")
    print("=" * 70)
    b_result = evaluate_classification(blunder_model, X_test, y2_test)
    print(f"Base rate (actual blunder frequency in test set): {b_result['base_rate']:.2%}")
    print(f"AUC = {b_result['auc']:.3f}")
    print(f"\nAt default threshold (0.5):")
    print(f"  precision={b_result['default']['precision']:.3f}  "
          f"recall={b_result['default']['recall']:.3f}  f1={b_result['default']['f1']:.3f}")
    print(f"At F1-optimal threshold ({b_result['tuned']['threshold']:.3f}):")
    print(f"  precision={b_result['tuned']['precision']:.3f}  "
          f"recall={b_result['tuned']['recall']:.3f}  f1={b_result['tuned']['f1']:.3f}")

    # --- Save reports -------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        {"model": q_result["model"], "task": "decision_quality",
         "MAE": q_result["MAE"], "RMSE": q_result["RMSE"], "R2": q_result["R2"]},
        {"model": b_result["model"], "task": "blunder (t=0.5)",
         "precision": b_result["default"]["precision"], "recall": b_result["default"]["recall"],
         "f1": b_result["default"]["f1"], "auc": b_result["auc"]},
        {"model": b_result["model"], "task": f"blunder (t={b_result['tuned']['threshold']:.2f}, tuned)",
         "precision": b_result["tuned"]["precision"], "recall": b_result["tuned"]["recall"],
         "f1": b_result["tuned"]["f1"], "auc": b_result["auc"]},
    ])
    summary.to_csv(REPORTS_DIR / "report_card_summary.csv", index=False)

    def _df_to_markdown_table(df: pd.DataFrame) -> str:
        # Avoids the optional 'tabulate' dependency that df.to_markdown() needs.
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, row in df.iterrows():
            cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    with open(REPORTS_DIR / "report_card_summary.md", "w") as f:
        f.write("# Model Report Card (saved models, no retraining)\n\n")
        f.write(f"Evaluated on: {len(X):,} moves / {n_games:,} games currently in "
                f"`data/processed/` (test split: {len(X_test):,} moves / "
                f"{game_id.iloc[test_idx].nunique():,} games)\n\n")
        f.write(_df_to_markdown_table(summary))
        f.write("\n\nSee `report_card_*.png` in `reports/figures/` for PR curve, "
                "threshold sweep, confusion matrices, and the regression scatter.\n")

    b_result["sweep"].to_csv(REPORTS_DIR / "report_card_threshold_sweep.csv", index=False)

    # Records the tuned cutoff for whatever model is CURRENTLY at
    # models/blunder_model.pkl - safe to write unconditionally, since this
    # never touches model weights, only the calibration metadata that goes
    # with them. Keeps prediction.load_blunder_threshold() in sync whenever
    # this script runs, without needing a full retrain via improve_pipeline.py.
    threshold_info = {
        "model": b_result["model"],
        "threshold": b_result["tuned"]["threshold"],
        "precision": b_result["tuned"]["precision"],
        "recall": b_result["tuned"]["recall"],
        "f1": b_result["tuned"]["f1"],
        "auc": b_result["auc"],
        "evaluated_on_n_moves": len(X),
        "evaluated_on_n_games": int(n_games),
        "note": "Use this threshold on predict_proba()[:, 1] instead of the "
                "sklearn default of 0.5 - see prediction.load_blunder_threshold().",
    }
    with open(MODELS_DIR / "blunder_threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)

    print("\nGenerating figures...")
    make_figures(q_result, b_result, y1_test, y2_test)

    print(f"\nSaved:\n  {REPORTS_DIR / 'report_card_summary.csv'}\n  {REPORTS_DIR / 'report_card_summary.md'}\n"
          f"  {REPORTS_DIR / 'report_card_threshold_sweep.csv'}\n  {MODELS_DIR / 'blunder_threshold.json'}\n"
          f"  {FIGURES_DIR}/report_card_*.png")


if __name__ == "__main__":
    main()
