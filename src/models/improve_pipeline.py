"""
Step 12f - Improvement Pipeline

Chains the steps that were previously separate manual calls into one command:

  1. Hyperparameter-tune every candidate (tuning.py's search functions -
     RandomForest/XGBoost regressors, LogReg/RandomForest/XGBoost classifiers)
  2. Score every tuned candidate AND the current baselines on the SAME held-out
     test split, using evaluate_saved_models.py's metrics (not each search's
     internal CV score - CV winners don't always win on the actual held-out
     test set, and that's the number that matters)
  3. For the blunder task specifically, sweep thresholds per candidate first -
     comparing candidates at their own best F1, not all at a shared default
     0.5, since a model with well-calibrated probabilities can lose to a
     worse model at 0.5 purely because of where its probabilities happen to
     sit (see the RandomForestClassifier F1=0.003-at-0.5-but-AUC=0.90 case
     this project already ran into)
  4. Save the single best regressor and single best classifier as labeled,
     size-suffixed files (e.g. models/blunder_model_5000games_331640moves.pkl)
     via model_registry.py - ALWAYS, regardless of dataset size, so nothing
     from this run is ever lost.
  5. Only PROMOTE those to the canonical models/decision_quality_model.pkl /
     models/blunder_model.pkl (the filenames prediction.py and the dashboard
     actually load) if this run's dataset meets the size run_pipeline.bat is
     actually configured for (N_GAMES in scripts/download_dataset.py).
     Otherwise the canonical files are left exactly as they were.

     This exists because the previous version of this script always
     overwrote the canonical files unconditionally, and a run against the
     local 300-game data/processed/ silently replaced a ~5,000-game model
     that had been placed there separately. See model_registry.py.
  6. If something was promoted, regenerate reports/ by calling
     evaluate_saved_models.main() against the newly-promoted models.

Deliberately does NOT touch data/processed/ or check what dataset produced
it - this is purely "given whatever data and candidate models exist, find
the best one, and only ship it if it's the right size." Run the
download/Stockfish/feature pipeline (run_pipeline.bat) separately first if
you need different data.

GPU: tuning.py defaults to USE_GPU=True for its two XGBoost searches. This
script forces it to False so the pipeline runs the same way on any machine;
edit FORCE_CPU below (or tuning.py's USE_GPU) if you have a working CUDA
setup and want the speedup.

Run:
    venv\\Scripts\\python.exe src\\models\\improve_pipeline.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tuning  # noqa: E402
import model_registry  # noqa: E402
from training import load_data, make_group_split  # noqa: E402
from evaluate_saved_models import evaluate_regression, evaluate_classification  # noqa: E402
import evaluate_saved_models  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

FORCE_CPU = True  # see module docstring


def build_baseline_candidates(X_train, y1_train, y2_train):
    """The un-tuned models training.py would produce, included so tuning has
    to actually beat them to be promoted - a tuned model that loses to the
    plain default isn't worth shipping just because it went through a search."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from xgboost import XGBRegressor, XGBClassifier

    neg, pos = (y2_train == 0).sum(), (y2_train == 1).sum()

    reg_candidates = {
        "LinearRegression (baseline)": LinearRegression(),
        "RandomForestRegressor (baseline)": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBRegressor (baseline)": XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    for model in reg_candidates.values():
        model.fit(X_train, y1_train)

    clf_candidates = {
        "LogisticRegression (baseline)": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForestClassifier (baseline)": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGBClassifier (baseline)": XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                   scale_pos_weight=neg / pos),
    }
    for model in clf_candidates.values():
        model.fit(X_train, y2_train)

    return reg_candidates, clf_candidates


def main():
    if FORCE_CPU:
        tuning.USE_GPU = False

    t0 = time.time()
    print("Loading data and building the group-based train/test split...")
    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]
    groups_train = game_id.iloc[train_idx]
    print(f"  {len(X_train):,} train / {len(X_test):,} test moves "
          f"({game_id.nunique():,} games total)")

    print("\n" + "=" * 70)
    print("Building baseline candidates (must be beaten to matter)")
    print("=" * 70)
    reg_candidates, clf_candidates = build_baseline_candidates(X_train, y1_train, y2_train)

    print("\n" + "=" * 70)
    print("Hyperparameter search - decision quality (this is the slow part)")
    print("=" * 70)
    best_rf_reg, best_xgb_reg = tuning.tune_decision_quality(X_train, y1_train, groups_train)
    reg_candidates["RandomForestRegressor (tuned)"] = best_rf_reg
    reg_candidates["XGBRegressor (tuned)"] = best_xgb_reg

    print("\n" + "=" * 70)
    print("Hyperparameter search - blunder classification")
    print("=" * 70)
    best_logreg, best_rf_clf, best_xgb_clf = tuning.tune_blunder(X_train, y2_train, groups_train)
    clf_candidates["LogisticRegression (tuned)"] = best_logreg
    clf_candidates["RandomForestClassifier (tuned)"] = best_rf_clf
    clf_candidates["XGBClassifier (tuned)"] = best_xgb_clf

    # --- Score every candidate on the SAME held-out test set ---------------
    print("\n" + "=" * 70)
    print("Scoring every candidate on the held-out test set")
    print("=" * 70)
    reg_scores = {}
    for name, model in reg_candidates.items():
        result = evaluate_regression(model, X_test, y1_test, y1_train)
        reg_scores[name] = result
        print(f"  [{name:35s}] MAE={result['MAE']:6.2f}  RMSE={result['RMSE']:6.2f}  R2={result['R2']:.4f}")

    clf_scores = {}
    for name, model in clf_candidates.items():
        result = evaluate_classification(model, X_test, y2_test)
        clf_scores[name] = result
        print(f"  [{name:35s}] AUC={result['auc']:.3f}  "
              f"F1@0.5={result['default']['f1']:.3f}  "
              f"F1@tuned({result['tuned']['threshold']:.2f})={result['tuned']['f1']:.3f}")

    # --- Pick winners: R2 for regression, tuned-threshold F1 for classification --
    best_reg_name = max(reg_scores, key=lambda k: reg_scores[k]["R2"])
    best_clf_name = max(clf_scores, key=lambda k: clf_scores[k]["tuned"]["f1"])

    print(f"\nWinner - decision quality: {best_reg_name} "
          f"(R2={reg_scores[best_reg_name]['R2']:.4f})")
    print(f"Winner - blunder:          {best_clf_name} "
          f"(F1={clf_scores[best_clf_name]['tuned']['f1']:.3f} "
          f"@ threshold={clf_scores[best_clf_name]['tuned']['threshold']:.3f})")

    # --- Always save a labeled, size-suffixed copy - never overwritten by a
    # later run with a different dataset size, and never silently overwrites
    # a bigger/different dataset's model under the shared canonical name -----
    n_games = game_id.nunique()
    n_moves = len(X)
    print(f"\nSaving labeled copies (dataset: {n_games:,} games / {n_moves:,} moves)...")
    versioned_reg_path = model_registry.save_versioned(
        reg_candidates[best_reg_name], "decision_quality_model", n_games, n_moves)
    versioned_clf_path = model_registry.save_versioned(
        clf_candidates[best_clf_name], "blunder_model", n_games, n_moves)
    print(f"  {versioned_reg_path.name}")
    print(f"  {versioned_clf_path.name}")

    threshold_info = {
        "model": best_clf_name,
        "threshold": clf_scores[best_clf_name]["tuned"]["threshold"],
        "precision": clf_scores[best_clf_name]["tuned"]["precision"],
        "recall": clf_scores[best_clf_name]["tuned"]["recall"],
        "f1": clf_scores[best_clf_name]["tuned"]["f1"],
        "auc": clf_scores[best_clf_name]["auc"],
        "trained_on_n_games": int(n_games),
        "trained_on_n_moves": int(n_moves),
        "note": "Use this threshold on predict_proba()[:, 1] instead of the "
                "sklearn default of 0.5 - see prediction.load_blunder_threshold().",
    }
    versioned_threshold_path = MODELS_DIR / (
        f"blunder_threshold_{model_registry.dataset_suffix(n_games, n_moves)}.json")
    with open(versioned_threshold_path, "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  {versioned_threshold_path.name}")

    # --- Only promote to the canonical production filenames if this run's
    # dataset actually meets the size run_pipeline.bat is configured for -----
    print(f"\nChecking against the configured target "
          f"({model_registry.TARGET_GAMES:,} games, scripts/download_dataset.py N_GAMES)...")
    promoted_reg = model_registry.promote_if_target_size(
        versioned_reg_path, "decision_quality_model.pkl", n_games)
    promoted_clf = model_registry.promote_if_target_size(
        versioned_clf_path, "blunder_model.pkl", n_games)
    if promoted_clf:
        shutil.copyfile(versioned_threshold_path, MODELS_DIR / "blunder_threshold.json")
        print("  Promoted matching threshold file -> models/blunder_threshold.json")

    if promoted_reg or promoted_clf:
        print("\n" + "=" * 70)
        print("Regenerating reports/ against the promoted models")
        print("=" * 70)
        evaluate_saved_models.main()
    else:
        print("\nNothing promoted - the canonical models/*.pkl files (and reports/) "
              "were left exactly as they were before this run. The labeled copies "
              "above are available for inspection but are not what prediction.py "
              "or the dashboard load by default.")

    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
