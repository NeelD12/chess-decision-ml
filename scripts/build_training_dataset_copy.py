import pandas as pd


def build_training_dataset(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["game_phase"], drop_first=True)

    feature_cols = [
        # player
        "player_elo", "opponent_elo", "rating_diff",
        # time
        "remaining_time", "time_spent", "time_remaining_ratio", "time_pressure",
        # position
        "material_balance", "piece_count", "legal_move_count",
        "king_safety", "doubled_pawns", "position_complexity", "eval_before",
        # behavioural
        "rolling_avg_cpl", "recent_accuracy", "eval_trend",
        "thinking_time_trend", "decision_consistency", "consecutive_errors",
    ]
    phase_cols = [c for c in df.columns if c.startswith("game_phase_")]
    feature_cols += phase_cols

    X = df[feature_cols].copy()

    y2 = df["is_blunder"]

    # y1: decision quality,
    # anything at/above 100 cpl floors at 0
    y1 = (100 - df["cpl"].clip(upper=100)).clip(lower=0)

    return X, y1, y2


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/moves_features.parquet")

    X, y1, y2 = build_training_dataset(df)

    print("X shape:", X.shape)
    print("\nNaN check per column:")
    print(X.isna().sum())

    print("\ny1 (decision quality) stats:")
    print(y1.describe())

    print("\ny2 (blunder) class balance:")
    print(y2.value_counts(normalize=True))

    X.to_parquet("data/processed/X_features.parquet")
    y1.to_frame("decision_quality").to_parquet("data/processed/y1_decision_quality.parquet")
    y2.to_frame("is_blunder").to_parquet("data/processed/y2_blunder.parquet")
    print("\nSaved X, y1, y2 to data/processed/")