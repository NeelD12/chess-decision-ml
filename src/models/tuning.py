"""
Step 12b - Hyperparameter Tuning

Takes the baseline models from training.py further: instead of one fixed set of
hyperparameters, this tries many combinations via randomized search and keeps
whichever combination scores best under cross-validation.

GPU note: scikit-learn (Random Forest, Logistic Regression) has NO GPU support at
all - that's a hard library limitation, not a config option. Only XGBoost can use a
GPU here, via USE_GPU below. At this dataset's size (tens of thousands of rows),
don't expect a dramatic speedup either way - GPU acceleration mostly matters at
millions of rows. What actually makes this take real time is the search itself:
N_ITER combinations x N_SPLITS cross-validation folds = N_ITER * N_SPLITS full
training runs per model, not one.

Cross-validation here uses GroupKFold, not plain KFold - same reasoning as the
train/test split in training.py: rows from the same game_id must never end up
split across a fold boundary during tuning either, or the search would silently
reward hyperparameters that let the model partially memorize a game.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

from training import load_data, make_group_split  # reuse Step 12's loading + split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

N_ITER = 30       # how many random hyperparameter combinations to try per model
N_SPLITS = 5      # cross-validation folds within the training set
RANDOM_STATE = 42

USE_GPU = True    # only affects the two XGBoost searches - see module docstring


def tune_decision_quality(X_train, y_train, groups_train):
    """
    Random Forest and XGBoost regressors, searched over the parameters that
    actually matter for tree-based models:
      - n_estimators: number of trees - more can help, but costs time
      - max_depth: how deep each tree can grow - deeper fits training data more
        closely, but risks overfitting the small dataset
      - min_samples_split / min_samples_leaf (RF): how many rows are required
        before a tree is allowed to split further - larger values = simpler,
        less overfit-prone trees
      - learning_rate (XGB only): how much each new tree corrects the ones
        before it - smaller values need more trees but generalize better
      - subsample / colsample_bytree (XGB only): fraction of rows/features each
        tree sees - less than 1.0 adds randomness that fights overfitting
    """
    cv = GroupKFold(n_splits=N_SPLITS)

    rf_param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        rf_param_grid, n_iter=N_ITER, cv=cv, scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=2,
    )
    rf_search.fit(X_train, y_train, groups=groups_train)
    print("Best RandomForestRegressor params:", rf_search.best_params_)
    print("Best RandomForestRegressor CV MAE:", -rf_search.best_score_)

    xgb_param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_kwargs = {"random_state": RANDOM_STATE, "n_jobs": -1}
    if USE_GPU:
        xgb_kwargs.update(tree_method="hist", device="cuda")

    xgb_search = RandomizedSearchCV(
        XGBRegressor(**xgb_kwargs),
        xgb_param_grid, n_iter=N_ITER, cv=cv, scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=1, verbose=2,  # n_jobs=1: the GPU work itself isn't parallelized across CV folds
    )
    xgb_search.fit(X_train, y_train, groups=groups_train)
    print("Best XGBRegressor params:", xgb_search.best_params_)
    print("Best XGBRegressor CV MAE:", -xgb_search.best_score_)

    return rf_search.best_estimator_, xgb_search.best_estimator_


def tune_blunder(X_train, y_train, groups_train):
    """
    Same idea for the classification side, scored on F1 (not accuracy - accuracy
    is misleading with rare blunders, see docs/step12_model_training.md).
    Logistic Regression only has one real knob worth searching here: C, the
    inverse regularization strength (smaller C = simpler/more regularized model).
    """
    cv = GroupKFold(n_splits=N_SPLITS)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    logreg_search = RandomizedSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        n_iter=5, cv=cv, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1, verbose=2,
    )
    logreg_search.fit(X_train, y_train, groups=groups_train)
    print("Best LogisticRegression params:", logreg_search.best_params_)
    print("Best LogisticRegression CV F1:", logreg_search.best_score_)

    rf_param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        rf_param_grid, n_iter=N_ITER, cv=cv, scoring="f1",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=2,
    )
    rf_search.fit(X_train, y_train, groups=groups_train)
    print("Best RandomForestClassifier params:", rf_search.best_params_)
    print("Best RandomForestClassifier CV F1:", rf_search.best_score_)

    xgb_param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_kwargs = {"random_state": RANDOM_STATE, "n_jobs": -1, "scale_pos_weight": pos_weight}
    if USE_GPU:
        xgb_kwargs.update(tree_method="hist", device="cuda")

    xgb_search = RandomizedSearchCV(
        XGBClassifier(**xgb_kwargs),
        xgb_param_grid, n_iter=N_ITER, cv=cv, scoring="f1",
        random_state=RANDOM_STATE, n_jobs=1, verbose=2,
    )
    xgb_search.fit(X_train, y_train, groups=groups_train)
    print("Best XGBClassifier params:", xgb_search.best_params_)
    print("Best XGBClassifier CV F1:", xgb_search.best_score_)

    return logreg_search.best_estimator_, rf_search.best_estimator_, xgb_search.best_estimator_


def evaluate_on_test(name, model, X_test, y_test, task):
    if task == "regression":
        preds = model.predict(X_test)
        print(f"[TEST] {name}: MAE={mean_absolute_error(y_test, preds):.2f}")
    else:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"[TEST] {name}: F1={f1_score(y_test, preds):.3f}  AUC={roc_auc_score(y_test, probs):.3f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]
    groups_train = game_id.iloc[train_idx]

    print("=" * 60, "\nTuning Decision Quality models\n", "=" * 60)
    best_rf_reg, best_xgb_reg = tune_decision_quality(X_train, y1_train, groups_train)
    evaluate_on_test("RandomForestRegressor (tuned)", best_rf_reg, X_test, y1_test, "regression")
    evaluate_on_test("XGBRegressor (tuned)", best_xgb_reg, X_test, y1_test, "regression")

    print("\n", "=" * 60, "\nTuning Blunder models\n", "=" * 60)
    best_logreg, best_rf_clf, best_xgb_clf = tune_blunder(X_train, y2_train, groups_train)
    evaluate_on_test("LogisticRegression (tuned)", best_logreg, X_test, y2_test, "classification")
    evaluate_on_test("RandomForestClassifier (tuned)", best_rf_clf, X_test, y2_test, "classification")
    evaluate_on_test("XGBClassifier (tuned)", best_xgb_clf, X_test, y2_test, "classification")

    # Saves every tuned model - decide which to keep as "final" after comparing
    # the [TEST] lines above, don't just take the CV winner blindly.
    joblib.dump(best_rf_reg, MODELS_DIR / "decision_quality_rf_tuned.pkl")
    joblib.dump(best_xgb_reg, MODELS_DIR / "decision_quality_xgb_tuned.pkl")
    joblib.dump(best_logreg, MODELS_DIR / "blunder_logreg_tuned.pkl")
    joblib.dump(best_rf_clf, MODELS_DIR / "blunder_rf_tuned.pkl")
    joblib.dump(best_xgb_clf, MODELS_DIR / "blunder_xgb_tuned.pkl")
    print("\nSaved all tuned models to models/")


if __name__ == "__main__":
    main()
