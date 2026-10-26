# Step 11 — Assembling the Real Training Dataset

**Status: ▶ FOLLOW NOW.** Steps 9, 9b, and 10 are done (`src/stockfish/engine.py`,
`src/features/feature_engineering.py`) — this step just selects and combines what
those already produced. No new engine calls, no new loops over FENs.

## What you actually have right now

From `data/processed/moves_features.parquet` (built by
`src/features/feature_engineering.py`), these real columns exist:

| group | columns |
|---|---|
| player | `player_elo`, `opponent_elo`, `rating_diff` |
| time | `remaining_time`, `base_time`, `increment`, `time_spent`, `time_remaining_ratio`, `time_pressure` |
| position | `material_balance`, `piece_count`, `legal_move_count`, `king_safety`, `doubled_pawns`, `game_phase`, `position_complexity` |
| engine | `eval_before`, `eval_after`, `eval_swing`, `played_best_move`, `cpl`, `move_quality` |
| behavioural | `is_blunder`, `is_good_move`, `rolling_avg_cpl`, `recent_accuracy`, `eval_trend`, `thinking_time_trend`, `decision_consistency`, `consecutive_errors` |

That's a real, populated feature set — not the placeholder list from the original
`docs/training_dataset.md`. This file supersedes that one now that the columns exist
for real.

## Building X

Pick your feature columns into one DataFrame. A reasonable starting set, leaving out
columns that would leak the answer or duplicate another column:

```
X_COLUMNS = [
    "player_elo", "opponent_elo", "rating_diff",
    "remaining_time", "time_spent", "time_remaining_ratio", "time_pressure",
    "material_balance", "piece_count", "legal_move_count",
    "king_safety", "doubled_pawns", "position_complexity",
    "eval_before", "position_complexity",
    "rolling_avg_cpl", "recent_accuracy", "eval_trend",
    "thinking_time_trend", "decision_consistency", "consecutive_errors",
]
```

**Watch out for leakage before you lock this list in:**
- `eval_after`, `eval_swing`, and `cpl` are all *outcomes* of the move that was played —
  a model trained with these in `X` would basically be told the answer. They belong in
  `y`, not `X`.
- `game_phase` is text (`"Opening"`/`"Middlegame"`/`"Endgame"`) — most models need it
  one-hot encoded (`pd.get_dummies`) before it can go into `X`.
- `move_quality` is also text and derived from `cpl` — it's closer to a label than a
  feature.

Decide this list yourself, but write down *why* you excluded anything you leave out —
future-you debugging a suspiciously perfect model will want to know you already
thought about leakage.

## Building y2 (blunder / no-blunder)

You already have it: `is_blunder` (built from `move_quality == "Blunder"` using your
10/50/100/300 CPL thresholds). No extra work needed — just confirm the class balance
isn't wildly skewed:

```python
df["is_blunder"].value_counts(normalize=True)
```

If blunders are rare (they usually are), that's expected and something to handle at
model-training time (Step 12), not a bug in this step.

## Building y1 (decision quality, 0-100)

This doesn't exist yet — you need to design it. It has to be a function of `cpl` where
0 CPL loss → near 100, and large CPL loss → near 0. A simple starting point:

```python
decision_quality = (100 - df["cpl"].clip(upper=100)).clip(lower=0)
```

...but that's one option, not the answer. Questions worth thinking through before you
commit to a formula:
- Should the scale be linear in CPL, or should it fall off faster for small mistakes
  than large ones (i.e. is losing 100 centipawns really "10x worse" than losing 10)?
- Should `decision_quality` account for `position_complexity` — is a mistake in a
  simple position worse than the same CPL loss in a complex one?

Whatever you land on, write the formula and reasoning in a short note (a paragraph at
the top of wherever you build `y1` is fine) — same principle as the CPL thresholds.

## One outstanding item from Step 9b

Looking at `classify_move_quality()` in `src/stockfish/engine.py` — you picked
10/50/100/300 as the CPL cutoffs, which is a completely reasonable set of numbers, but
`docs/label_thresholds.md` (referenced in `docs/GUIDE.md`) was never written. Worth a
quick paragraph on why you chose those specific numbers before moving on — it's the
kind of thing a reviewer (or later-you) will ask about.

## Before Step 12

Once `X`, `y1`, `y2` exist and look right (`.describe()`, check for `NaN`s introduced
by rolling windows on the first move of each game — `min_periods=1` should have handled
that, but verify), that's the natural checkpoint to stop and check in before touching
model code.
