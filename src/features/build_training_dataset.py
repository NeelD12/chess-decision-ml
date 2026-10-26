"""
Step 11 - Assemble the real training dataset (X, y1, y2) from the feature-engineered
dataset produced by feature_engineering.py.

X excludes eval_after / eval_swing / cpl on purpose - those are outcomes of the move
being predicted, not information available before the decision (see
docs/step11_training_dataset.md).
"""
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_training_dataset(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["game_phase"], drop_first=True)

    feature_cols = [
        # player
        "player_elo", "opponent_elo", "rating_diff",
        # time
        "remaining_time", "time_spent", "time_remaining_ratio", "time_pressure",
        # position
        "material_balance", "piece_count", "legal_move_count",
        "king_safety", "doubled_pawns", "position_complexity",
        # engine (eval_before only - it's known BEFORE the move, no leakage)
        "eval_before",
        # behavioural
        "rolling_avg_cpl", "recent_accuracy", "eval_trend",
        "thinking_time_trend", "decision_consistency", "consecutive_errors",
    ]
    phase_cols = [c for c in df.columns if c.startswith("game_phase_")]
    feature_cols += phase_cols
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y2 = df["is_blunder"]

    # y1: decision quality, linear in cpl - a bigger centipawn loss costs you
    # quality points at a constant rate, capped so anything at/above 100 cpl
    # floors out at 0 rather than going negative.
    y1 = (100 - df["cpl"].clip(upper=100)).clip(lower=0)

    return X, y1, y2


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "moves_features.parquet")

    X, y1, y2 = build_training_dataset(df)

    print("X shape:", X.shape)
    print("\nNaN check per column:")
    print(X.isna().sum())

    print("\ny1 (decision quality) stats:")
    print(y1.describe())

    print("\ny2 (blunder) class balance:")
    print(y2.value_counts(normalize=True))

    X.to_parquet(PROCESSED_DIR / "X_features.parquet")
    y1.to_frame("decision_quality").to_parquet(PROCESSED_DIR / "y1_decision_quality.parquet")
    y2.to_frame("is_blunder").to_parquet(PROCESSED_DIR / "y2_blunder.parquet")
    print(f"\nSaved X, y1, y2 to {PROCESSED_DIR}")
