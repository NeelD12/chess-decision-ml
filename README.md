# Chess Decision Quality & Blunder Prediction

An explainable-ML project that studies **human decision quality in chess** — specifically
how time pressure, position complexity, and rating affect the chance of a blunder.

Pipeline (Steps 1-12 and 15 built and working; Step 14/SHAP not started):

```
Chess PGN Dataset
  -> Dataset Validation & Cleaning
  -> PGN Parsing / Move Extraction
  -> Move-level Dataset
  -> Stockfish Analysis -> Engine-Based Labels -> Feature Engineering
  -> Training Dataset Assembly (X, y1, y2)
  -> Baseline Models + Hyperparameter Tuning
  -> Results Dashboard (incl. live inference on uploaded games)
  -> (not started, by design) Explainable AI (SHAP)
```

Run the whole thing (download through training) in one command:
`run_pipeline.bat` — see [Running the full pipeline](#running-the-full-pipeline) below.

## Project status

- [x] Dev environment (Python 3.11+, venv, dependencies)
- [x] Git / GitHub set up
- [x] PGN sample downloaded (currently 5,000 games — see `scripts/download_dataset.py`)
- [x] PGN parser built (game-level + move-level, including FEN and clock data)
- [x] Data exported to CSV/Parquet
- [x] Initial validation & cleaning documented
- [x] Step 9 — Stockfish analysis (`src/stockfish/engine.py`, `run_full_analysis.py`)
- [x] Step 9b — CPL + move-quality labels (thresholds documented pending — see `docs/GUIDE.md`)
- [x] Step 10 — Feature engineering, all 5 groups (`src/features/feature_engineering.py`)
- [x] Step 11 (code) — `X` / `y1` / `y2` assembly (`src/features/build_training_dataset.py`)
- [x] Step 12 — baseline models (`src/models/training.py`) + GPU hyperparameter tuning (`src/models/tuning.py`)
- [x] Step 12d — inference on saved models, including live inference on an uploaded PGN (`src/models/prediction.py`)
- [x] Step 15 — results dashboard with model comparison, "Try a Prediction," and "Upload a Game" tabs (`dashboard/app.py`)
- [ ] Step 14 — SHAP explainability — **not started yet, by design** — see `docs/step14_explainable_ai.md`

## Folder structure

```
chess-decision-ml/
|-- data/
|   |-- raw/          # original downloaded PGNs (not committed to git)
|   |-- interim/      # partially processed data (not committed to git)
|   `-- processed/    # final CSV/Parquet datasets (not committed to git)
|-- notebooks/        # exploration notebooks
|-- src/
|   |-- data/         # pgn_parser.py, cleaning.py
|   |-- stockfish/    # engine.py, run_full_analysis.py (checkpointed, resumable)
|   |-- features/     # feature_engineering.py, build_training_dataset.py
|   `-- models/       # training.py, tuning.py, evaluation.py, prediction.py
|-- models/           # saved trained models (.pkl, gitignored - regenerate via the pipeline)
|-- reports/          # generated figures + metrics_summary.csv/.md - committed on purpose, these ARE the paper's results
|-- dashboard/        # Streamlit app - results + live inference
|-- configs/          # config files
|-- scripts/          # one-off helper scripts (e.g. dataset download)
|-- logs/             # run_pipeline.bat logs (gitignored)
|-- requirements.txt
|-- setup.bat         # one-click environment setup for Windows
|-- run_pipeline.bat  # one-click full pipeline: download -> ... -> training
`-- .gitignore
```

## Setup (Windows)

1. Clone this repo:
   ```
   git clone https://github.com/<your-username>/chess-decision-ml.git
   cd chess-decision-ml
   ```
2. Double-click `setup.bat`, or run it from a terminal:
   ```
   setup.bat
   ```
   This installs Python 3.11 if needed, creates a virtual environment (`venv/`),
   and installs everything in `requirements.txt`.
3. Activate the environment whenever you come back to work on the project:
   ```
   venv\Scripts\activate
   ```
4. Launch Jupyter Lab to explore notebooks:
   ```
   jupyter lab
   ```

## Running the full pipeline

Once `setup.bat` has run once, `run_pipeline.bat` runs everything from downloading
the dataset through baseline model training, in order, logging every step's output
to both the console and a timestamped file under `logs/`:

```
run_pipeline.bat            REM download -> Stockfish analysis -> features -> training
run_pipeline.bat tune       REM same, plus GPU hyperparameter tuning at the end
```

**Step 2 (Stockfish analysis) is the slow one** — hours at the current 5,000-game
target, since every move needs 2 engine calls. It's safe to close the terminal or
`Ctrl+C` partway through: `src/stockfish/run_full_analysis.py` checkpoints every 500
rows to `data/processed/moves_analyzed.parquet` and automatically resumes from there
next time, instead of starting over.

Once training's done, explore the results:
```
venv\Scripts\streamlit run dashboard\app.py
```

## Dataset

Games are streamed from **Lichess's open database**
(`database.lichess.org`, standard rated games), rather than a generic PGN dump you'd
find elsewhere. `scripts/download_dataset.py` opens a streaming connection to that
month's file, decompresses it on the fly, and keeps only games that contain per-move
clock (`%clk`) annotations — stopping as soon as it has `N_GAMES` of them (currently
5,000 - see the comment in that file for the timing tradeoff behind that number),
without ever downloading the full multi-gigabyte file.

This was a deliberate change from "download any public PGN file": many public PGN dumps
strip clock annotations entirely, and since this project's core question is about
**decision quality under time pressure**, clock data isn't optional. Filtering straight
from the source guarantees every kept game has it.

See `data/README.md` for exact source, game count, and field documentation.

## Why move-level, not game-level?

Each game contains dozens of individual decisions. The ML dataset represents **one row per
move**, not one row per game, because we're predicting the quality of a single decision, not
the outcome of a whole game.

## What's next (Neel — read this)

Steps 9 through 12 and 15 are done - real code, tested end to end, including a live
dashboard. Only **Step 14 (SHAP / explainability)** is left before this matches the
full roadmap.

- **`docs/step14_explainable_ai.md`** — start here for what's left.
- **`docs/GUIDE.md`** — overall status + the full per-step file index.
- **`dashboard/app.py`** — run it (`venv\Scripts\streamlit run dashboard\app.py`) and
  try the "Upload a Game" tab with one of your own PGNs (with ratings + clock times) -
  it runs live Stockfish analysis + feature engineering + prediction on data the model
  has genuinely never seen, not just a row from the training set.
- **`notebooks/03_stockfish_analysis.ipynb`** and **`notebooks/04_model_experiments.ipynb`**
  — the original scaffolds for Steps 9-12, superseded by the real code in `src/` now,
  kept for the explanations/resources.
