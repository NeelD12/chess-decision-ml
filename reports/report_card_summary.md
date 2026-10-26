# Model Report Card (saved models, no retraining)

Evaluated on: 331,640 moves / 5,000 games currently in `data/processed/` (test split: 65,355 moves / 1,000 games)

| model | task | MAE | RMSE | R2 | precision | recall | f1 | auc |
|---|---|---|---|---|---|---|---|---|
| XGBRegressor | decision_quality | 29.748 | 35.404 | 0.121 | nan | nan | nan | nan |
| XGBClassifier | blunder (t=0.5) | nan | nan | nan | 0.146 | 0.577 | 0.233 | 0.788 |
| XGBClassifier | blunder (t=0.71, tuned) | nan | nan | nan | 0.227 | 0.349 | 0.275 | 0.788 |

See `report_card_*.png` in `reports/figures/` for PR curve, threshold sweep, confusion matrices, and the regression scatter.
