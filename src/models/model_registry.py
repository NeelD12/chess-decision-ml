"""
Step 12g - Model Registry

Keeps "how many games/moves was this actually trained on" attached to the
filename itself, and stops any pipeline from silently overwriting the
canonical production models (models/decision_quality_model.pkl,
models/blunder_model.pkl) with a model trained on a smaller/different
dataset than run_pipeline.bat is actually configured for.

This exists because that exact mistake already happened once: a pipeline
run against the local 300-game data/processed/ silently replaced Neel's real
~5,000-game models, because both trainings wrote to the same filename with
no size check in between. Every save through this module is labeled with
its dataset size, and only a model trained on (approximately) the
configured target size is allowed to become the canonical production file.
"""
import shutil
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from download_dataset import N_GAMES as TARGET_GAMES  # noqa: E402

# download_dataset.py can legitimately land a bit under N_GAMES if it hits
# MAX_SCANNED first - this is how far short a run is still allowed to be and
# still count as "the real dataset" for promotion purposes.
PROMOTION_TOLERANCE = 0.9


def dataset_suffix(n_games: int, n_moves: int) -> str:
    return f"{n_games}games_{n_moves}moves"


def save_versioned(model, base_name: str, n_games: int, n_moves: int) -> Path:
    """
    Always saves a labeled, permanent copy - e.g.
    models/decision_quality_model_300games_20163moves.pkl - that a later run
    with a different dataset size can never collide with or overwrite.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{base_name}_{dataset_suffix(n_games, n_moves)}.pkl"
    joblib.dump(model, path)
    return path


def promote_if_target_size(versioned_path: Path, canonical_name: str, n_games: int) -> bool:
    """
    Copies versioned_path -> models/<canonical_name> ONLY if n_games meets the
    configured target (scripts/download_dataset.py's N_GAMES, within
    PROMOTION_TOLERANCE). Returns whether it promoted.

    Refusing below that size means a stray small-data run can save its own
    labeled copy for inspection without ever being able to silently downgrade
    the production file every other script and the dashboard actually load.
    """
    if n_games < TARGET_GAMES * PROMOTION_TOLERANCE:
        print(f"  NOT promoting to models/{canonical_name}: this run used {n_games:,} games, "
              f"below the configured target of {TARGET_GAMES:,} games (N_GAMES in "
              f"scripts/download_dataset.py). The labeled copy at "
              f"models/{versioned_path.name} was saved, but models/{canonical_name} "
              f"was left untouched.")
        return False
    shutil.copyfile(versioned_path, MODELS_DIR / canonical_name)
    print(f"  Promoted models/{versioned_path.name} -> models/{canonical_name} "
          f"({n_games:,} games meets the {TARGET_GAMES:,}-game target).")
    return True
