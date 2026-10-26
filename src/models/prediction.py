"""
Step 12d - Inference

Loads an already-trained, saved model and predicts on new rows, instead of
retraining every time a prediction is needed. training.py (baselines) and
tuning.py (GPU hyperparameter search) both save their models to models/*.pkl -
this module is the one place that loads them back for actual use.

Prefers a tuned model (from tuning.py) over a baseline model (from training.py)
when both exist, since the tuned one is the better-performing "final" model -
but falls back to the baseline gracefully if tuning hasn't been run yet.
"""

import io
import json
import sys
from pathlib import Path

import chess
import chess.pgn
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
BLUNDER_THRESHOLD_PATH = MODELS_DIR / "blunder_threshold.json"

for _sub in ("data", "stockfish", "features"):
    sys.path.insert(0, str(PROJECT_ROOT / "src" / _sub))
from pgn_parser import parse_game  # noqa: E402
from engine import get_engine, analyze_position, compute_cpl, classify_move_quality  # noqa: E402
from feature_engineering import (  # noqa: E402
    add_player_features, add_time_features, add_position_features,
    add_engine_features, add_behavioural_features,
)
from build_training_dataset import build_training_dataset  # noqa: E402

# Preference order: best tuned model first, baseline last as a fallback.
QUALITY_MODEL_CANDIDATES = [
    MODELS_DIR / "decision_quality_xgb_tuned.pkl",
    MODELS_DIR / "decision_quality_rf_tuned.pkl",
    MODELS_DIR / "decision_quality_model.pkl",
]
BLUNDER_MODEL_CANDIDATES = [
    MODELS_DIR / "blunder_xgb_tuned.pkl",
    MODELS_DIR / "blunder_rf_tuned.pkl",
    MODELS_DIR / "blunder_logreg_tuned.pkl",
    MODELS_DIR / "blunder_model.pkl",
]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_models():
    """
    Returns (quality_model, blunder_model, quality_path, blunder_path).
    Raises FileNotFoundError with a clear next step if nothing has been
    trained yet - run training.py (and optionally tuning.py) first.
    """
    quality_path = _first_existing(QUALITY_MODEL_CANDIDATES)
    blunder_path = _first_existing(BLUNDER_MODEL_CANDIDATES)

    if quality_path is None or blunder_path is None:
        raise FileNotFoundError(
            "No saved models found in models/. Run:\n"
            "  python src/models/training.py\n"
            "(and optionally python src/models/tuning.py for the tuned versions)\n"
            "before running inference."
        )

    quality_model = joblib.load(quality_path)
    blunder_model = joblib.load(blunder_path)
    return quality_model, blunder_model, quality_path, blunder_path


def load_blunder_threshold() -> float:
    """
    Returns the F1-optimal cutoff written by improve_pipeline.py, or 0.5 if
    that pipeline has never been run. A hard-coded 0.5 is not meaningful for
    a ~5% base-rate event (see docs/step12_model_training.md) - this is the
    one place that decision is made, so every caller stays consistent
    automatically once the tuned threshold exists.
    """
    if BLUNDER_THRESHOLD_PATH.exists():
        with open(BLUNDER_THRESHOLD_PATH) as f:
            return json.load(f)["threshold"]
    return 0.5


def predict(quality_model, blunder_model, X_row: pd.DataFrame) -> dict:
    """
    X_row: a single-row DataFrame with the exact same columns (same order
    doesn't matter, but same names) as the X built in
    src/features/build_training_dataset.py.

    Returns predicted decision quality (0-100), blunder probability (0-1),
    and a hard blunder_flag using the tuned threshold from
    models/blunder_threshold.json (falls back to 0.5 if that file doesn't
    exist yet - i.e. improve_pipeline.py hasn't been run).
    """
    decision_quality = float(quality_model.predict(X_row)[0])
    blunder_probability = float(blunder_model.predict_proba(X_row)[:, 1][0])
    threshold = load_blunder_threshold()
    return {
        "decision_quality": decision_quality,
        "blunder_probability": blunder_probability,
        "blunder_flag": blunder_probability >= threshold,
        "blunder_threshold_used": threshold,
    }


def analyze_uploaded_pgn(pgn_text: str, expected_columns) -> pd.DataFrame:
    """
    Takes a raw PGN (as text - e.g. an uploaded file's contents) for ONE game
    and runs it through the same pipeline the training data went through:
    parse -> Stockfish analysis -> feature engineering -> build X/y1/y2.

    This is real "new data" inference, not just picking a row that was already
    in the training set - every feature is computed fresh for this game.

    expected_columns: the trained model's X.columns (e.g. from
    data/processed/X_features.parquet). A short or unusual game might not
    contain every game_phase seen during training (e.g. no Endgame moves),
    so the resulting columns are reindexed to match exactly, filling any
    missing ones with 0 - without this, predict() would fail or silently
    misalign columns against a differently-shaped model input.

    Raises ValueError with a human-readable reason if the PGN can't be used
    (unparseable, missing Elo headers, or missing clock annotations - the
    model needs all of these, same as the training data did).

    Returns a DataFrame with one row per move: game context (move number,
    player, played move), actual y1/y2 (if computable), and the model's
    predicted decision quality / blunder probability for that move.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse this as a PGN game.")

    rows = list(parse_game(game))
    if not rows:
        raise ValueError("No moves found in this game.")

    moves_df = pd.DataFrame(rows)

    if moves_df["white_elo"].isna().all() or moves_df["black_elo"].isna().all():
        raise ValueError(
            "This PGN is missing WhiteElo/BlackElo headers - the model needs player "
            "ratings. Export from Lichess/chess.com with player ratings included."
        )
    if moves_df["remaining_time"].isna().all():
        raise ValueError(
            "This PGN has no clock annotations (%clk comments) - the model needs "
            "per-move clock data. On Lichess, use 'Download with clock times'."
        )

    engine = get_engine()
    eval_befores, eval_afters, best_moves, cpls, qualities = [], [], [], [], []
    try:
        for _, row in moves_df.iterrows():
            board = chess.Board(row["fen_before"])
            side_to_move = board.turn

            eval_before, best_move = analyze_position(board, engine)
            move = board.parse_san(row["played_move"])
            board.push(move)
            eval_after, _ = analyze_position(board, engine)

            cpl = compute_cpl(eval_before, eval_after, side_to_move)
            quality = classify_move_quality(cpl)

            eval_befores.append(eval_before)
            eval_afters.append(eval_after)
            best_moves.append(best_move.uci() if best_move else None)
            cpls.append(cpl)
            qualities.append(quality)
    finally:
        engine.quit()

    moves_df["eval_before"] = eval_befores
    moves_df["eval_after"] = eval_afters
    moves_df["best_move"] = best_moves
    moves_df["cpl"] = cpls
    moves_df["move_quality"] = qualities

    moves_df = add_player_features(moves_df)
    moves_df = add_time_features(moves_df)
    moves_df = add_position_features(moves_df)
    moves_df = add_engine_features(moves_df)
    moves_df = add_behavioural_features(moves_df)

    X_new, y1_new, y2_new = build_training_dataset(moves_df)
    X_new = X_new.reindex(columns=expected_columns, fill_value=0)

    quality_model, blunder_model, _, _ = load_models()
    predicted_quality = quality_model.predict(X_new)
    predicted_blunder_prob = blunder_model.predict_proba(X_new)[:, 1]
    threshold = load_blunder_threshold()

    result = moves_df[["move_number", "player", "played_move"]].copy()
    result["actual_decision_quality"] = y1_new.values
    result["predicted_decision_quality"] = predicted_quality
    result["actual_blunder"] = y2_new.values.astype(bool)
    result["predicted_blunder_probability"] = predicted_blunder_prob
    result["predicted_blunder_flag"] = predicted_blunder_prob >= threshold

    # add_time_features()/add_behavioural_features() sort by (game_id, player,
    # move_number) internally to compute rolling stats correctly - restore
    # chronological move order here since that's what a human reading the
    # table actually wants to see.
    return result.sort_values("move_number").reset_index(drop=True)


if __name__ == "__main__":
    quality_model, blunder_model, quality_path, blunder_path = load_models()
    print(f"Loaded decision-quality model from: {quality_path.relative_to(PROJECT_ROOT)}")
    print(f"Loaded blunder model from:          {blunder_path.relative_to(PROJECT_ROOT)}")

    X = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "X_features.parquet")
    row = X.iloc[[0]]
    result = predict(quality_model, blunder_model, row)
    print(f"\nSmoke test - prediction for row 0:")
    print(f"  Decision quality:   {result['decision_quality']:.1f} / 100")
    print(f"  Blunder probability: {result['blunder_probability']:.1%}")
