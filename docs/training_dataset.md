# Step 11 - Training Dataset (conceptual)

This document is the roadmap's Step 11 deliverable: describing the **shape** of the
eventual training dataset. It is conceptual only - **no model is trained here**, and no
Stockfish/engine analysis has been run yet. That starts at Step 12 (Stockfish integration),
which is intentionally out of scope for this stage.

## What we have right now (Steps 1-10, already built)

`data/processed/moves_clean.parquet` currently has **20,163 rows** (one per move/decision)
from **300 real games**, pulled from Lichess's July 2026 rated-games database and cleaned
with zero rows dropped (see `data/README.md` for the full data quality report). Each row
looks like:

| column | meaning |
|---|---|
| `game_id` | which game this move belongs to |
| `move_number` | which move of the game |
| `player` | "white" or "black" |
| `white_elo` / `black_elo` | player ratings |
| `result` | game result (`1-0`, `0-1`, `1/2-1/2`) |
| `time_control` | e.g. `180+2` |
| `speed_category` | Bullet / Blitz / Rapid / Classical |
| `event`, `date`, `opening` | game metadata |
| `fen_before` | board position before the move was played |
| `played_move` | the move the human actually played |
| `remaining_time` | seconds left on the mover's clock after the move, if known |

This is real, already-collected data - not a projection.

## What Step 11 conceptually adds (not built yet)

Once Stockfish analysis (Step 9) and feature engineering (Step 10) exist, each row above
would be extended with derived features, conceptually:

```
X = player features + time features + position features + engine features + behavioural features

y1 = decision quality        (e.g. a 0-100 score derived from centipawn loss)
y2 = blunder / no-blunder    (a 0/1 label, e.g. centipawn loss above some threshold)
```

Worked example of a single future training row (illustrative, not real data):

```
Elo               = 1800
Remaining Time    = 12 sec
Move Number       = 38
Material Balance  = -1
Engine Eval       = +0.7
Previous CPL      = 120
Complexity        = 0.82
...

Target:
Decision Quality  = 34/100
Blunder           = 1
```

## Feature groups planned for Step 10 (not yet implemented)

- **Player features**: player Elo, opponent Elo, rating difference, historical performance
- **Time features**: remaining time, time spent on this move, average thinking time, time
  pressure indicator, time-remaining ratio, recent time consumption
- **Position features**: material balance, piece count, legal move count, king safety,
  piece activity, pawn structure, game phase, position complexity
- **Engine features** (need Stockfish, Step 9): evaluation before/after the move, evaluation
  swing, centipawn loss, best-move difference, tactical difficulty
- **Behavioural features**: previous CPL, rolling average CPL, recent accuracy, consecutive
  errors, evaluation trend, recent thinking-time trend, decision consistency

The final feature set will be finalized after EDA and correlation/importance analysis, as
the roadmap specifies - this is a plan, not a locked schema.

## Explicitly not done in this stage

- No Stockfish analysis has been run (Step 9)
- No engine-based labels exist yet (Excellent / Good / Inaccuracy / Mistake / Blunder
  thresholds still need to be researched and documented)
- No feature engineering code has been written (Step 10)
- **No model training** (Step 12) - by design, per the task scope for this stage
