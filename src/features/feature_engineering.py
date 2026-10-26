import numpy as np
import pandas as pd
import chess


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["white_elo"] = pd.to_numeric(df["white_elo"], errors="coerce")
    df["black_elo"] = pd.to_numeric(df["black_elo"], errors="coerce")

    is_white = df["player"] == "white"

    df["player_elo"] = df["white_elo"].where(is_white, df["black_elo"])
    df["opponent_elo"] = df["black_elo"].where(is_white, df["white_elo"])
    df["rating_diff"] = df["player_elo"] - df["opponent_elo"]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["remaining_time"] = pd.to_numeric(df["remaining_time"], errors="coerce")

    split_tc = df["time_control"].str.split("+", expand=True)
    df["base_time"] = pd.to_numeric(split_tc[0], errors="coerce")
    df["increment"] = pd.to_numeric(split_tc[1], errors="coerce").fillna(0)

    df = df.sort_values(["game_id", "player", "move_number"])

    prev_remaining = df.groupby(["game_id", "player"])["remaining_time"].shift(1)
    prev_remaining = prev_remaining.fillna(df["base_time"])

    df["time_spent"] = (prev_remaining + df["increment"] - df["remaining_time"]).clip(lower=0)
    df["time_remaining_ratio"] = np.where(df["base_time"] > 0, df["remaining_time"] / df["base_time"], 0.0)
    df["time_pressure"] = (df["time_remaining_ratio"] < 0.1).astype(int)

    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                     chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    material_balances, piece_counts, legal_move_counts = [], [], []
    king_safeties, doubled_pawns_list, game_phases, complexities = [], [], [], []

    for fen in df["fen_before"]:
        board = chess.Board(fen)

        material = 0
        for pt, val in piece_values.items():
            material += val * len(board.pieces(pt, chess.WHITE))
            material -= val * len(board.pieces(pt, chess.BLACK))
        material_balances.append(material)

        piece_counts.append(len(board.piece_map()))
        legal_move_counts.append(len(list(board.legal_moves)))

        king_square = board.king(board.turn)
        unsafe = 0
        if king_square is not None:
            for adj in board.attacks(king_square):
                if board.is_attacked_by(not board.turn, adj):
                    unsafe += 1
        king_safeties.append(unsafe)

        pawns = board.pieces(chess.PAWN, board.turn)
        files = [chess.square_file(sq) for sq in pawns]
        doubled_pawns_list.append(len(files) - len(set(files)))

        non_pawn_material = sum(
            piece_values[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        )
        if non_pawn_material >= 28:
            phase = "Opening"
        elif non_pawn_material >= 14:
            phase = "Middlegame"
        else:
            phase = "Endgame"
        game_phases.append(phase)

        tension = sum(
            1 for sq, piece in board.piece_map().items()
            if board.is_attacked_by(not piece.color, sq)
        )
        complexities.append(tension)

    df["material_balance"] = material_balances
    df["piece_count"] = piece_counts
    df["legal_move_count"] = legal_move_counts
    df["king_safety"] = king_safeties
    df["doubled_pawns"] = doubled_pawns_list
    df["game_phase"] = game_phases
    df["position_complexity"] = complexities

    return df


def add_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["eval_swing"] = df["eval_after"] - df["eval_before"]

    played_best = []
    for _, row in df.iterrows():
        board = chess.Board(row["fen_before"])
        played_uci = board.parse_san(row["played_move"]).uci()
        played_best.append(int(played_uci == row["best_move"]))
    df["played_best_move"] = played_best

    return df


def add_behavioural_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["game_id", "player", "move_number"])

    df["is_blunder"] = (df["move_quality"] == "Blunder").astype(int)
    df["is_good_move"] = df["move_quality"].isin(["Excellent", "Good"]).astype(int)

    grouped = df.groupby(["game_id", "player"])

    # FIXED (leakage): .rolling() includes the CURRENT row by default, so these
    # were partly built from the current move's own cpl / eval_swing - exactly
    # what is_blunder and decision_quality are derived from. That's why the
    # blunder classifier was scoring a suspicious 1.000 across every metric.
    # .shift(1) before .rolling() makes each feature look only at moves BEFORE
    # this one - "recent history going into this decision," not including it.
    # The first move of a game has no prior history, so the shift produces NaN
    # there - filled with 0 as a neutral "no history yet" placeholder.
    df["rolling_avg_cpl"] = grouped["cpl"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)
    df["recent_accuracy"] = grouped["is_good_move"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)
    df["eval_trend"] = grouped["eval_swing"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)
    df["thinking_time_trend"] = grouped["time_spent"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).fillna(0)
    df["decision_consistency"] = grouped["cpl"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    ).fillna(0)

    # FIXED (leakage): this loop used to fold the CURRENT move's own is_blunder
    # into the streak before recording it, so consecutive_errors partly encoded
    # the answer for this row. Now it records the streak built from moves
    # BEFORE this one, then updates the streak with the current move afterward
    # (for the next iteration) - so this means "blunders in a row leading up
    # to this decision," not "including this decision."
    df["consecutive_errors"] = 0
    for _, group in grouped:
        streak, streaks = 0, []
        for is_blunder in group["is_blunder"]:
            streaks.append(streak)
            streak = streak + 1 if is_blunder else 0
        df.loc[group.index, "consecutive_errors"] = streaks

    return df


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet("data/processed/moves_analyzed.parquet")

    print("Adding player features...")
    df = add_player_features(df)

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding position features (loops every row, may take a bit)...")
    df = add_position_features(df)

    print("Adding engine features...")
    df = add_engine_features(df)

    print("Adding behavioural features...")
    df = add_behavioural_features(df)

    print("\n--- Position features ---")
    print(df[["material_balance", "piece_count", "legal_move_count",
               "king_safety", "doubled_pawns", "position_complexity"]].describe())
    print(df["game_phase"].value_counts())

    print("\n--- Engine features ---")
    print(df[["eval_swing", "played_best_move"]].describe())

    print("\n--- Behavioural features ---")
    print(df[["rolling_avg_cpl", "recent_accuracy", "consecutive_errors",
               "eval_trend", "thinking_time_trend", "decision_consistency"]].describe())

    df.to_parquet("data/processed/moves_features.parquet")
    print("\nSaved to data/processed/moves_features.parquet")