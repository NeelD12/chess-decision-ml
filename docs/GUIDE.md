# Project Guide: What to Follow, What to Defer

This is the master status + directions file. Each step now has its own detailed file
(`docs/stepNN_*.md`) — this page is the index and the current status.

Legend:
- ✅ **DONE**
- ▶ **FOLLOW NOW** — start this, in order
- ⏳ **DEFER** — don't start yet, and why
- ℹ️ **READ ONLY** — a design note, not an action

---

## ✅ Step 9 — Stockfish Analysis — DONE

Built in `src/stockfish/engine.py` (`evaluate_position`, `get_best_move`,
`analyze_game`), verified end-to-end on a 5-row sample in `test_analysis.py`.

**One portability issue worth fixing:** `STOCKFISH_PATH` is hardcoded to an absolute
path on one specific machine (`C:\Users\ASUS\...`). It works today because it's your
own laptop, but it'll break the moment this code runs anywhere else (your mentor's
machine, a future reinstall, anyone else who clones the repo) — and setup.bat already
puts `stockfish` on PATH, so `chess.engine.SimpleEngine.popen_uci("stockfish")` (no
absolute path needed) should just work everywhere. Worth changing the default.

## ✅ Step 9b — Engine-Based Labels — DONE (mostly)

`compute_cpl` and `classify_move_quality` are implemented with thresholds
10 / 50 / 100 / 300 (Excellent / Good / Inaccuracy / Mistake / Blunder).

**Still outstanding:** `docs/label_thresholds.md` was never written. The numbers you
picked are reasonable, but the roadmap specifically asks for the *reasoning* to be
documented, not just the numbers - a paragraph comparing them to how Lichess/chess.com
define their own thresholds, and why you landed where you did. Details in
`docs/step11_training_dataset.md`.

## ✅ Step 10 — Feature Engineering — DONE

All five feature groups built in `src/features/feature_engineering.py`: player, time,
position, engine, behavioural. Real column list documented in
`docs/step11_training_dataset.md`.

---

## ▶ Step 11 (code) — Assemble the Real Training Dataset — FOLLOW NOW

**→ Full details: [`docs/step11_training_dataset.md`](step11_training_dataset.md)**

Build `X` from the real feature columns Step 10 produced, `y2` from the `is_blunder`
column you already have, and design a `y1` (decision quality 0-100) formula from `cpl`.
Watch for leakage: `eval_after`, `eval_swing`, and `cpl` are outcomes of the move, not
inputs to predicting it.

---

## ⏳ Step 12 — Model Training (Both Models) — DEFER until Step 11 is checked in

**→ Full details: [`docs/step12_model_training.md`](step12_model_training.md)**

Two models sharing one `X`: a regressor for decision quality (`y1`), a classifier for
blunder probability (`y2`). Baselines first (Linear/Logistic Regression → Random
Forest → XGBoost), split by `game_id` not by row, watch for class imbalance on the
blunder side.

---

## ℹ️ Step 13 — One System, Not Two Projects — READ ONLY

**→ Full details: [`docs/step13_unified_pipeline.md`](step13_unified_pipeline.md)**

Design note for when you build Step 12: shared feature-building code and identical
train/test splits for both models, not two disconnected mini-projects.

---

## ⏳ Step 14 — Explainable AI (SHAP) — DEFER until Step 12 exists

**→ Full details: [`docs/step14_explainable_ai.md`](step14_explainable_ai.md)**

`shap.TreeExplainer` on whichever model ends up tree-based. Nothing to explain until
there's a trained model to explain.

---

## ⏳ Step 15 — Dashboard (Streamlit) — DEFER until Steps 12 and 14 exist

**→ Full details: [`docs/step15_dashboard.md`](step15_dashboard.md)**

Roadmap's own words: "not our immediate priority — we need the data and model working
first."

---

## Quick-reference: right now, in order

1. Fix the hardcoded Stockfish path in `src/stockfish/engine.py` (quick, low-risk)
2. Write `docs/label_thresholds.md` — document your Step 9b reasoning
3. Build `X`, `y1`, `y2` per `docs/step11_training_dataset.md`
4. Stop. Check in before touching Step 12.

Other references still around: `docs/NEXT_STEPS.md` (the original resource links for
Steps 9-10, now historical since those are done) and `docs/training_dataset.md` (the
original *conceptual* Step 11 write-up from before real features existed — superseded
by `docs/step11_training_dataset.md`, kept for history).
