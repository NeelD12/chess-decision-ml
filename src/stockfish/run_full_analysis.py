"""
Runs Stockfish analysis (engine.py's analyze_position/compute_cpl/
classify_move_quality) across the FULL cleaned dataset, not just a small
sample - this is the long-running step (hours at 5,000+ games), so it:

  - Checkpoints to data/processed/moves_analyzed.parquet every
    CHECKPOINT_EVERY rows, so a crash, sleep, or closed terminal doesn't
    lose completed work.
  - Resumes automatically: if moves_analyzed.parquet already has N rows,
    it picks up at row N instead of starting over.
  - Prints progress with a live rows/sec rate and ETA, and logs individual
    row failures instead of crashing the whole run on one bad position.

Usage (from the project root, with the venv activated):
    python src/stockfish/run_full_analysis.py

Safe to Ctrl+C and rerun later - it'll resume from the last checkpoint.
"""

import sys
import time
import traceback
from pathlib import Path

import chess
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import get_engine, analyze_position, compute_cpl, classify_move_quality  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "moves_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "moves_analyzed.parquet"

CHECKPOINT_EVERY = 500
PROGRESS_EVERY = 100


def _load_checkpoint(total: int):
    """Returns (n_done, eval_befores, eval_afters, best_moves, cpls, qualities)."""
    if not OUTPUT_PATH.exists():
        return 0, [], [], [], [], []

    done_df = pd.read_parquet(OUTPUT_PATH)
    n_done = len(done_df)
    if n_done >= total:
        return n_done, None, None, None, None, None  # signals "already complete"

    print(f"Resuming from checkpoint: {n_done}/{total} already done")
    return (
        n_done,
        done_df["eval_before"].tolist(),
        done_df["eval_after"].tolist(),
        done_df["best_move"].tolist(),
        done_df["cpl"].tolist(),
        done_df["move_quality"].tolist(),
    )


def _save(moves_df, eval_befores, eval_afters, best_moves, cpls, qualities):
    n = len(eval_befores)
    result = moves_df.iloc[:n].copy()
    result["eval_before"] = eval_befores
    result["eval_after"] = eval_afters
    result["best_move"] = best_moves
    result["cpl"] = cpls
    result["move_quality"] = qualities
    result.to_parquet(OUTPUT_PATH)
    return n


def main():
    moves_df = pd.read_parquet(INPUT_PATH)
    total = len(moves_df)

    n_done, eval_befores, eval_afters, best_moves, cpls, qualities = _load_checkpoint(total)
    if eval_befores is None:
        print(f"Already complete: {n_done}/{total} rows in {OUTPUT_PATH}")
        return

    remaining = moves_df.iloc[n_done:]
    engine = get_engine()
    start = time.time()

    try:
        for offset, (_, row) in enumerate(remaining.iterrows()):
            i = n_done + offset
            try:
                board = chess.Board(row["fen_before"])
                side_to_move = board.turn

                eval_before, best_move = analyze_position(board, engine)

                move = board.parse_san(row["played_move"])
                board.push(move)

                eval_after, _ = analyze_position(board, engine)
                cpl = compute_cpl(eval_before, eval_after, side_to_move)
                quality = classify_move_quality(cpl)
            except chess.engine.EngineTerminatedError:
                raise  # fatal - the engine process died, no point continuing
            except Exception as row_err:
                print(f"  Row {i} failed ({row_err!r}), filling with None")
                eval_before = eval_after = cpl = None
                best_move = None
                quality = None

            eval_befores.append(eval_before)
            eval_afters.append(eval_after)
            best_moves.append(best_move.uci() if best_move else None)
            cpls.append(cpl)
            qualities.append(quality)

            if (i + 1) % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                rate = (offset + 1) / elapsed
                eta_min = (total - i - 1) / rate / 60 if rate > 0 else float("inf")
                print(f"Row {i + 1}/{total}  ({rate:.2f} rows/s, ETA {eta_min:.1f} min)")

            if (i + 1) % CHECKPOINT_EVERY == 0:
                _save(moves_df, eval_befores, eval_afters, best_moves, cpls, qualities)
                print(f"  Checkpoint saved at row {i + 1}")

    except Exception:
        print("FATAL ERROR - saving what's done so far before exiting:")
        traceback.print_exc()
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    n = _save(moves_df, eval_befores, eval_afters, best_moves, cpls, qualities)
    print(f"\nSaved {n}/{total} rows to {OUTPUT_PATH}")
    if n < total:
        print("INCOMPLETE - rerun this script to resume from where it stopped.")
    else:
        print("COMPLETE.")


if __name__ == "__main__":
    main()
