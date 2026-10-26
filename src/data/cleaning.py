"""
Validates and cleans the move-level dataset produced by pgn_parser.py.

Checks performed (per the roadmap's Step 8 checklist):
  - missing ratings (white_elo / black_elo)
  - missing clock values (remaining_time)
  - duplicate games (same game_id parsed more than once)
  - incomplete games (result == "*", i.e. aborted/unfinished)
  - abnormal / unparseable time controls
  - missing results
  - unrealistic Elo values

It classifies each game's time control into Bullet / Blitz / Rapid /
Classical using Lichess's own speed thresholds (based on estimated total
game duration = base_seconds + 40 * increment_seconds), because the
relationship between time pressure and decision quality is expected to
differ a lot between, say, bullet and classical games.

Writes a human-readable report to data/processed/data_quality_report.md
and a cleaned dataset to data/processed/moves_clean.parquet.
"""

from pathlib import Path

import pandas as pd


def classify_time_control(time_control: str):
    """'180+2' -> 'Blitz', using Lichess's own speed thresholds."""
    if not isinstance(time_control, str) or "+" not in time_control:
        return None
    try:
        base_str, increment_str = time_control.split("+")
        base = int(base_str)
        increment = int(increment_str)
    except ValueError:
        return None

    estimated_total = base + 40 * increment
    if estimated_total < 30:
        return "UltraBullet"
    if estimated_total < 180:
        return "Bullet"
    if estimated_total < 480:
        return "Blitz"
    if estimated_total < 1500:
        return "Rapid"
    return "Classical"


def build_report(moves_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    games_df = moves_df.drop_duplicates(subset="game_id").copy()
    games_df["speed_category"] = games_df["time_control"].apply(classify_time_control)

    n_games = len(games_df)
    n_moves = len(moves_df)

    missing_white_elo = games_df["white_elo"].isna().sum()
    missing_black_elo = games_df["black_elo"].isna().sum()
    missing_result = games_df["result"].isna().sum()
    aborted_games = (games_df["result"] == "*").sum()
    unparseable_time_control = games_df["speed_category"].isna().sum()

    dup_move_rows = moves_df.duplicated(subset=["game_id", "move_number"]).sum()

    moves_missing_clock = moves_df["remaining_time"].isna().sum()
    pct_missing_clock = 100 * moves_missing_clock / n_moves if n_moves else 0

    elo_numeric = pd.to_numeric(games_df["white_elo"], errors="coerce").dropna()
    elo_numeric = pd.concat([elo_numeric, pd.to_numeric(games_df["black_elo"], errors="coerce").dropna()])
    unrealistic_elo = ((elo_numeric < 400) | (elo_numeric > 3500)).sum()

    speed_counts = games_df["speed_category"].value_counts(dropna=False).to_dict()

    lines = [
        "# Data Quality Report",
        "",
        f"- Games parsed: {n_games}",
        f"- Moves (decisions) parsed: {n_moves}",
        "",
        "## Missing / invalid fields",
        f"- Games missing White Elo: {missing_white_elo}",
        f"- Games missing Black Elo: {missing_black_elo}",
        f"- Games missing Result: {missing_result}",
        f"- Aborted / incomplete games (Result == '*'): {aborted_games}",
        f"- Games with unparseable / abnormal TimeControl: {unparseable_time_control}",
        f"- Duplicate (game_id, move_number) rows: {dup_move_rows}",
        f"- Moves missing clock data: {moves_missing_clock} ({pct_missing_clock:.1f}%)",
        f"- Unrealistic Elo values (<400 or >3500): {unrealistic_elo}",
        "",
        "## Games by speed category",
    ]
    for category, count in speed_counts.items():
        lines.append(f"- {category}: {count}")

    report_text = "\n".join(lines)
    return games_df, report_text
#this is my code 

def clean(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows/games with problems that would break downstream steps."""
    cleaned = moves_df.drop_duplicates(subset=["game_id", "move_number"]).copy()

    # drop games that never finished (result == "*") - no valid outcome to learn from
    valid_game_ids = cleaned.loc[cleaned["result"] != "*", "game_id"].unique()
    cleaned = cleaned[cleaned["game_id"].isin(valid_game_ids)]

    cleaned["speed_category"] = cleaned["time_control"].apply(classify_time_control)
    cleaned = cleaned[cleaned["speed_category"].notna()]

    return cleaned


def main():
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    moves_df = pd.read_parquet(processed_dir / "moves.parquet")

    _, report_text = build_report(moves_df)
    report_path = processed_dir / "data_quality_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved report to {report_path}")

    cleaned_df = clean(moves_df)
    clean_path = processed_dir / "moves_clean.parquet"
    cleaned_df.to_parquet(clean_path, index=False)
    print(f"Saved cleaned dataset ({len(cleaned_df)} rows) to {clean_path}")


if __name__ == "__main__":
    main()
