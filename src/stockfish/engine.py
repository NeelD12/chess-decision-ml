import chess
import chess.engine
import pandas as pd

# FIXED (portability): was hardcoded to one machine's absolute install path.
# setup.bat puts Stockfish on PATH for everyone, so "stockfish" alone works
# on any machine that ran setup.bat - no per-machine path needed.
STOCKFISH_PATH = "stockfish"

DEFAULT_DEPTH = 12


def get_engine(path: str = STOCKFISH_PATH) -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(path)


def evaluate_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = DEFAULT_DEPTH,
) -> int:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    return score.score(mate_score=10000)


def get_best_move(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = DEFAULT_DEPTH,
) -> chess.Move:
    result = engine.play(board, chess.engine.Limit(depth=depth))
    return result.move


def analyze_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = DEFAULT_DEPTH,
):
    """
    IMPROVED (speed): one engine.analyse() call returns both the evaluation
    AND the best line (info["pv"]) for a position - engine.analyse() already
    computes the best move as a side effect of finding the score. Using this
    instead of a separate evaluate_position() + get_best_move() call pair
    cuts Stockfish calls per move from 3 to 2 (~33% faster on a full run),
    with identical accuracy since it's the same search at the same depth.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=10000)
    best_move = info["pv"][0] if info.get("pv") else None
    return score, best_move


def compute_cpl(eval_before: int, eval_after: int, side_to_move: bool) -> int:
    if side_to_move == chess.WHITE:
        loss = eval_before - eval_after
    else:
        loss = eval_after - eval_before
    return max(0, loss)


def classify_move_quality(cpl: int) -> str:
    if cpl < 10:
        return "Excellent"
    elif cpl < 50:
        return "Good"
    elif cpl < 100:
        return "Inaccuracy"
    elif cpl < 300:
        return "Mistake"
    else:
        return "Blunder"
    
def analyze_game(moves_df: pd.DataFrame, engine: chess.engine.SimpleEngine) -> pd.DataFrame:
    eval_befores = []
    eval_afters = []
    best_moves = []
    cpls = []
    qualities = []

    total = len(moves_df)

    for i, (_, row) in enumerate(moves_df.iterrows()):
        if i % 500 == 0:
            print(f"Processing row {i}/{total}...")

        board = chess.Board(row["fen_before"])
        side_to_move = board.turn

        eval_before, best_move = analyze_position(board, engine)

        move = board.parse_san(row["played_move"])
        board.push(move)

        eval_after = evaluate_position(board, engine)
        cpl = compute_cpl(eval_before, eval_after, side_to_move)
        quality = classify_move_quality(cpl)

        eval_befores.append(eval_before)
        eval_afters.append(eval_after)
        best_moves.append(best_move.uci())
        cpls.append(cpl)
        qualities.append(quality)

    result_df = moves_df.copy()
    result_df["eval_before"] = eval_befores
    result_df["eval_after"] = eval_afters
    result_df["best_move"] = best_moves
    result_df["cpl"] = cpls
    result_df["move_quality"] = qualities

    return result_df