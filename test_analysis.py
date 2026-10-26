import sys
sys.path.append("src/stockfish")

import pandas as pd
from engine import get_engine, analyze_game

moves_df = pd.read_parquet("data/processed/moves_clean.parquet")
sample_df = moves_df.head(5)

print("Testing on first 5 moves...")

engine = get_engine()
result = analyze_game(sample_df, engine)
engine.quit()

print(result[["fen_before", "played_move", "eval_before", "eval_after", "cpl", "move_quality"]])