"""
Parses PGN games into a move-level dataset.

For every move we record:
  - game-level metadata (players, ratings, result, time control, ...)
  - the board position BEFORE the move was played (as a FEN string) -
    this is the position the player was actually deciding on
  - the move that was played
  - the player's remaining clock time after playing the move, if the
    PGN includes clock annotations ("%clk" comments)

One PGN game with 40 moves becomes 40 rows - one row per decision -
because the ML dataset needs to be move-level, not game-level.
"""

import re
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import pandas as pd

CLOCK_RE = re.compile(r"\[%clk\s+(\d+):(\d{2}):(\d{2})\]")


def clock_to_seconds(comment: str):
    """Turn a '[%clk 0:01:23]' PGN comment into 83 (seconds). Returns None if absent."""
    match = CLOCK_RE.search(comment)
    if not match:
        return None
    hours, minutes, seconds = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def extract_game_id(headers: chess.pgn.Headers) -> str:
    """Lichess stores the game id as the last part of the Site URL."""
    site = headers.get("Site", "")
    if site:
        return site.rstrip("/").split("/")[-1]
    return headers.get("Event", "unknown")


def parse_game(game: chess.pgn.Game) -> Iterator[dict]:
    """Yield one dict per move (decision) in a single parsed game."""
    headers = game.headers
    game_id = extract_game_id(headers)
    white_elo = headers.get("WhiteElo")
    black_elo = headers.get("BlackElo")
    result = headers.get("Result")
    time_control = headers.get("TimeControl")
    event = headers.get("Event")
    date = headers.get("UTCDate") or headers.get("Date")
    opening = headers.get("Opening")

    board = game.board()
    move_number = 0
    node = game
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        player = "white" if board.turn == chess.WHITE else "black"
        move_number += 1

        yield {
            "game_id": game_id,
            "move_number": move_number,
            "player": player,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": result,
            "time_control": time_control,
            "event": event,
            "date": date,
            "opening": opening,
            "fen_before": board.fen(),
            "played_move": board.san(move),
            "remaining_time": clock_to_seconds(next_node.comment),
        }

        board.push(move)
        node = next_node


def parse_pgn_file(path) -> Iterator[dict]:
    """Read a .pgn file game by game and yield move-level dict rows."""
    with open(path, encoding="utf-8") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield from parse_game(game)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "sample_games.pgn"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(parse_pgn_file(input_path))
    df = pd.DataFrame(rows)

    csv_path = output_dir / "moves.csv"
    parquet_path = output_dir / "moves.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    n_games = df["game_id"].nunique() if not df.empty else 0
    print(f"Parsed {len(df)} moves from {n_games} games.")
    print(f"Saved to:\n  {csv_path}\n  {parquet_path}")


if __name__ == "__main__":
    main()
