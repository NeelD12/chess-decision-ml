import sys
sys.path.append("src/stockfish")

import pandas as pd
import time
from engine import get_engine, analyze_game

print("Loading dataset...")
moves_df = pd.read_parquet("data/processed/moves_clean.parquet")
print(f"Loaded {len(moves_df)} rows")

engine = get_engine()
start = time.time()

result = analyze_game(moves_df, engine)

engine.quit()
elapsed = time.time() - start
print(f"Done in {elapsed/60:.1f} minutes")

result.to_parquet("data/processed/moves_analyzed.parquet")
print("Saved to data/processed/moves_analyzed.parquet")