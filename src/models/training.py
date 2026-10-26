"""
Step 12 - Model Training

Trains two independent models on the same feature set (X):
  - Model A: decision quality (y1) -> regression
  - Model B: blunder probability (y2) -> classification

Both use the SAME train/test split (Step 13 - one system, not two disconnected
projects), split by game_id rather than by row so moves from the same game never
end up on both sides (see docs/step12_model_training.md for why that matters).

Baselines are tried in increasing complexity - we don't assume XGBoost wins by
default, we compare and see.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """Load X, y1, y2 from Step 11's output, plus game_id (only used for splitting -
    it's never a feature the model sees)."""
    X = pd.read_parquet(PROCESSED_DIR / "X_features.parquet")
    y1 = pd.read_parquet(PROCESSED_DIR / "y1_decision_quality.parquet")["decision_quality"]
    y2 = pd.read_parquet(PROCESSED_DIR / "y2_blunder.parquet")["is_blunder"]

    moves_df = pd.read_parquet(PROCESSED_DIR / "moves_features.parquet")
    game_id = moves_df["game_id"]

    assert X.isna().sum().sum() == 0, "X still has NaNs - fix Step 11 before training"

    return X, y1, y2, game_id


def make_group_split(X, groups, test_size=0.2, random_state=42):
    """
    One split, reused for both targets. GroupShuffleSplit keeps every row from
    the same game_id entirely on one side - a plain random row split would let
    the model partially 'see' a game during training that it's tested on later.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, groups=groups))
    return train_idx, test_idx


def train_decision_quality_models(X_train, X_test, y_train, y_test):
    """Model A - regression. Baselines: Linear -> Random Forest -> XGBoost."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=1),
        "XGBRegressor": XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"[Decision Quality] {name:22s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:.3f}")

    return results


def train_blunder_models(X_train, X_test, y_train, y_test):
    """
    Model B - classification. Blunders are the minority class, so plain accuracy
    would be misleading (predicting 'no blunder' every time scores high) -
    we use class_weight="balanced" (sklearn) / scale_pos_weight (xgboost) and
    report precision/recall/F1/AUC instead of accuracy.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced", verbose=1
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, scale_pos_weight=neg / pos
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        results[name] = {"model": model, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
        print(f"[Blunder] {name:22s} precision={precision:.3f}  recall={recall:.3f}  "
              f"F1={f1:.3f}  AUC={auc:.3f}")
        print(f"  confusion matrix:\n{confusion_matrix(y_test, preds)}")

    return results


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]

    print("=" * 60, "\nDecision Quality (regression)\n", "=" * 60)
    quality_results = train_decision_quality_models(X_train, X_test, y1_train, y1_test)

    print("\n", "=" * 60, "\nBlunder Probability (classification)\n", "=" * 60)
    blunder_results = train_blunder_models(X_train, X_test, y2_train, y2_test)

    # Picks the best of each group by one metric - look at ALL the printed
    # numbers above before trusting this pick, not just the one it optimizes for.
    best_quality = min(quality_results, key=lambda k: quality_results[k]["MAE"])
    best_blunder = max(blunder_results, key=lambda k: blunder_results[k]["f1"])

    joblib.dump(quality_results[best_quality]["model"], MODELS_DIR / "decision_quality_model.pkl")
    joblib.dump(blunder_results[best_blunder]["model"], MODELS_DIR / "blunder_model.pkl")

    print(f"\nSaved best decision-quality model ({best_quality}) -> models/decision_quality_model.pkl")
    print(f"Saved best blunder model ({best_blunder}) -> models/blunder_model.pkl")


if __name__ == "__main__":
    main()