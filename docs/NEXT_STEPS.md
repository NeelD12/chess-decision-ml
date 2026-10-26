# Next Steps for Neel - Steps 9, 10, and beyond

Everything up through Step 11 (conceptual training-dataset shape) is already built for
you in this repo: environment, PGN parser, cleaning, and `docs/training_dataset.md`.

**Starting here, you write the code yourself.** This file is a guide, not a script -
it tells you what each step needs to do and where to learn how, but the actual `.py`
files are yours to write. That's the point: you'll understand the project a lot better
if you build this part hands-on instead of being handed finished code.

Work through these in order. Don't skip to feature engineering before Stockfish works -
each step depends on the one before it.

---

## Step 9 - Stockfish Analysis

**Goal:** for every position in `data/processed/moves_clean.parquet` (the `fen_before`
column), ask Stockfish "how good is this position, and what's the best move here?"

**What the file should do** (`src/stockfish/engine.py`):
1. Open a connection to the Stockfish engine (you already verified this works - see the
   test command in the terminal history, using `chess.engine.SimpleEngine.popen_uci`).
2. Write a function that takes a FEN string and returns Stockfish's evaluation (a score
   in centipawns, or "mate in N") at some fixed depth or time limit.
3. Loop over your cleaned dataset's `fen_before` values, call that function for each one,
   and save the evaluation alongside the row (e.g. a new `eval_before` column).
4. Do the same for the position **after** the human's move was played, so you can later
   compare "best possible outcome" vs. "what actually happened."

**Things to research and decide yourself before coding:**
- How much time/depth to give Stockfish per position (there's a real tradeoff: deeper
  = more accurate but slower - with ~300 games x ~40 moves each, that's over 10,000
  positions, so this matters).
- What format to store an evaluation in when it's "mate in N" instead of a centipawn
  number (python-chess's `PovScore` object handles both - read how).
- Whether to analyze every move or only a sample first, to sanity-check your code before
  running it on the whole dataset (small-first is a habit worth keeping from the
  earlier steps).

**Resources:**
- python-chess engine module docs (this is the API you'll use):
  https://python-chess.readthedocs.io/en/latest/engine.html
- Stockfish UCI protocol reference (how the engine communicates under the hood):
  https://official-stockfish.github.io/docs/stockfish-wiki/UCI-&-Commands-Stockfish.html
- Chess Programming Wiki on centipawns (what the numbers actually mean):
  https://www.chessprogramming.org/Centipawns

---

## Step 9b - Engine-Based Labels (Centipawn Loss & Blunder Thresholds)

**Goal:** turn raw evaluations into a "how good was this decision" number, and a
blunder yes/no label.

**Centipawn Loss (CPL)** is the difference between the evaluation of the best move and
the evaluation of the move actually played, from the mover's perspective. A move with
CPL near 0 was (near-)optimal; a move with high CPL threw away an advantage.

**What you need to research and document (the roadmap explicitly says these thresholds
should be researched, not guessed):**
- What CPL ranges other chess-analysis tools use to define Excellent / Good /
  Inaccuracy / Mistake / Blunder. Lichess and chess.com both publish their own
  definitions - compare them, and note in your own doc which you're adopting and why.
- Whether the same CPL threshold should apply regardless of position complexity (a
  100 CPL loss in a totally lost position is very different from a 100 CPL loss in an
  equal position).

**Resources:**
- Lichess's own explanation of their accuracy/CPL model (blog):
  https://lichess.org/page/accuracy
- Search "centipawn loss blunder threshold" for a few independent write-ups and compare
  them before picking your own thresholds - and write your reasoning down in
  `docs/label_thresholds.md` once you decide.

---

## Step 10 - Feature Engineering

**Goal:** build `src/features/feature_engineering.py`, which takes the cleaned +
Stockfish-analyzed dataset and produces the actual numeric columns a model can train on.

Build these one group at a time, and check each one with `.describe()` / a histogram
before moving to the next group - it's much easier to debug one feature group than five
at once.

- **Player features**: player Elo, opponent Elo, rating difference. (You already have
  the raw Elo columns - this group is mostly arithmetic.)
- **Time features**: remaining time (already have it), time spent on this move
  (difference between consecutive `remaining_time` values for the same player - think
  about the increment!), time-pressure indicator (e.g. remaining time below some
  threshold), time-remaining ratio.
- **Position features**: material balance, piece count, legal move count - all
  computable directly from a `chess.Board()` built from `fen_before` using python-chess's
  board methods (`board.legal_moves`, `board.piece_map()`, etc.)
- **Engine features**: evaluation before/after (from Step 9), evaluation swing,
  centipawn loss (from Step 9b).
- **Behavioural features**: rolling average CPL, recent accuracy, consecutive errors -
  these need pandas `.rolling()` / `.expanding()` grouped by `game_id` and `player`.

**Resources:**
- python-chess Board API (for position features):
  https://python-chess.readthedocs.io/en/latest/core.html#chess.Board
- pandas rolling-window docs (for behavioural features):
  https://pandas.pydata.org/docs/reference/window.html
- Kaggle's free "Feature Engineering" micro-course (good general background):
  https://www.kaggle.com/learn/feature-engineering

---

## Step 11 (code) - Assembling the Real Training Dataset

Once Steps 9 and 10 produce real columns, revisit `docs/training_dataset.md` - the `X`
and `y1`/`y2` layout described there conceptually now becomes actual code: select your
feature columns into `X`, and build `y1` (decision quality) and `y2` (blunder label)
from your Step 9b work.

---

## Step 12+ - Model Training (later, with your mentor's sign-off)

Not started yet, on purpose - the roadmap wants Step 12 kicked off deliberately, not
squeezed in. When you get there:

- scikit-learn docs: https://scikit-learn.org/stable/
- XGBoost docs: https://xgboost.readthedocs.io/
- SHAP docs + example notebooks (for the explainability stage after training):
  https://shap.readthedocs.io/

---

## General habits worth keeping from Steps 1-11

- **Prototype small, validate, scale** - test any new script on ~10-20 rows before
  running it on the full dataset.
- **Commit after each working milestone** - once Stockfish analysis works on a small
  sample, commit it, *then* scale up.
- **Write down decisions you make** (thresholds, time limits, feature definitions) in a
  markdown file next to the code, the way `docs/training_dataset.md` documents this
  stage - future-you will thank present-you.
