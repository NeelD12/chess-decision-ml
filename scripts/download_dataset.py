"""
Download a small sample of rated Lichess games that include per-move
clock data, WITHOUT downloading the entire multi-gigabyte monthly file.

Why this approach (instead of grabbing a random PGN file off the internet):
  Lichess publishes one huge compressed file per month containing every
  rated game played that month (tens of GB). We open a streaming HTTP
  connection to that file, decompress it a little at a time, and read
  it one game at a time with python-chess. As soon as we've kept
  N_GAMES games that actually contain clock annotations ("%clk" comments)
  we stop and close the connection - so we never download the whole file.

  We keep only games WITH clock data because this project's core question
  (decision quality under time pressure) needs it - a dataset without
  clocks would be useless for that part of the roadmap.

Usage (from the project root, with the venv activated):
    python scripts/download_dataset.py
"""

import io
from pathlib import Path

import chess.pgn
import requests
import zstandard as zstd

# A recent month's standard-rated games dump. Change the date to pull a
# different month - see https://database.lichess.org/ for what's available.
DATABASE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-07.pgn.zst"

# Scaling up per the roadmap's "prototype small -> validate -> scale" plan
# (300 -> 5,000 -> 10,000 -> 50,000+). 300 games is fully validated (leakage
# fixed, pipeline confirmed working end to end on two machines).
#
# 5,000 games is the target for the paper's real results - the largest size
# that's actually feasible on a laptop in one sitting: at ~0.071s/move (after
# the engine.py speedup that merged 3 Stockfish calls/move down to 2),
# 5,000 games (~336,000 moves) is roughly 6.5-7 hours of Stockfish analysis.
# 10,000 (~13 hours) and 50,000+ (multiple days) are documented here as
# future options, not the current default - run src/stockfish/run_full_analysis.py
# overnight or across a couple of sessions (it checkpoints and resumes).
N_GAMES = 5000          # how many games (with clock data) to keep
MAX_SCANNED = 200000    # safety cap so the script can't run forever

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_games.pgn"


def has_clock_data(game: chess.pgn.Game) -> bool:
    node = game
    while node.variations:
        node = node.variations[0]
        if "%clk" in node.comment:
            return True
    return False


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {DATABASE_URL} ...")
    response = requests.get(DATABASE_URL, stream=True)
    response.raise_for_status()

    kept = 0
    scanned = 0

    try:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
                while kept < N_GAMES and scanned < MAX_SCANNED:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        print("Reached the end of the stream before collecting enough games.")
                        break

                    scanned += 1
                    if has_clock_data(game):
                        print(game, file=out)
                        print(file=out)
                        kept += 1

                    if scanned % 500 == 0:
                        print(f"  scanned {scanned} games, kept {kept} with clock data...")
    finally:
        response.close()  # stops the download - we never fetch the rest of the file

    print(f"\nDone. Scanned {scanned} games, saved {kept} games with clock data to:")
    print(f"  {OUTPUT_PATH}")
    if kept < N_GAMES:
        print(f"WARNING: only found {kept}/{N_GAMES} games with clock data before the scan cap.")


if __name__ == "__main__":
    main()
