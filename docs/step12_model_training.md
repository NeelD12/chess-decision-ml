# Step 12 — Model Training (Both Models)

**Status: ⏳ Do not start until Step 11 (`X`, `y1`, `y2`) is built and checked in with
your mentor.** This file exists so you know what's coming and can read ahead — not as
a green light to start coding it yet.

There are **two separate models**, sharing the same `X`:

```
                 ┌──→ Model A: Decision Quality (regression) → predicts y1
X (features) ────┤
                 └──→ Model B: Blunder Probability (classification) → predicts y2
```

They're two models internally, but one system conceptually (see Step 13) — same input
features, different targets, trained and evaluated separately.

## Model A — Decision Quality (regression)

Predicts `y1`, the 0-100 score you designed in Step 11.

**Baselines to try, in order, per the roadmap (don't jump straight to XGBoost):**
1. `sklearn.linear_model.LinearRegression` — your simplest baseline. If a fancier model
   can't beat this by much, that's useful information, not a failure.
2. `sklearn.ensemble.RandomForestRegressor`
3. `xgboost.XGBRegressor`

**Metrics to report for each:** MAE and RMSE (both interpretable in "quality score
units"), plus R². Look at *which rows* have the worst errors, not just the aggregate
number — is the model bad specifically at endgames, or at high time pressure?

## Model B — Blunder Probability (classification)

Predicts `y2` (`is_blunder`, 0/1).

**Baselines to try, in the same spirit:**
1. `sklearn.linear_model.LogisticRegression`
2. `sklearn.ensemble.RandomForestClassifier`
3. `xgboost.XGBClassifier`

**Class imbalance is the thing to watch here.** Blunders are a minority class — check
`y2.value_counts(normalize=True)` before training anything. If it's heavily skewed
(e.g. 5% blunders), plain accuracy will look great by just predicting "no blunder"
every time and be useless. Use precision/recall/F1 and a confusion matrix instead of
accuracy alone, and look into `class_weight="balanced"` (available on both
LogisticRegression and RandomForestClassifier) as a first lever to try.

## The split that matters more than usual here: by game, not by row

Rows from the same game are correlated (same players, same position sequence, same
tim“e-control conditions). If you split train/test *randomly by row*, moves from the
same game can end up on both sides of the split, and the model can effectively
"memorize" a game it partially saw during training. Split by `game_id` instead:

```python
from sklearn.model_selection import train_test_split

game_ids = df["game_id"].unique()
train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)

train_df = df[df["game_id"].isin(train_ids)]
test_df = df[df["game_id"].isin(test_ids)]
```

This one detail matters more for the validity of your results than which algorithm you
pick.

## After baselines

```
Baseline → Evaluation → Feature improvements → Hyperparameter tuning → Final model
```

Same loop for both models, run independently. Don't tune hyperparameters before you've
looked hard at *which* predictions are wrong and why — that's usually more informative
than a grid search at this stage.

## Resources

- scikit-learn model evaluation guide: https://scikit-learn.org/stable/modules/model_evaluation.html
- XGBoost docs: https://xgboost.readthedocs.io/
- scikit-learn `train_test_split` and `GroupShuffleSplit` (an alternative to the manual
  game_id split above, worth comparing): https://scikit-learn.org/stable/modules/cross_validation.html#group-cv
