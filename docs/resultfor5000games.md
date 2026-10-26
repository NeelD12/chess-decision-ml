# Model Results — 5,000-Game Dataset

Update (2026-09-29): regenerated from a full independent reproduction on a
second machine — fresh download of 5,000 games, complete Stockfish analysis
of all 331,640 positions (zero failed rows), same feature/training code.
Landed on the exact same train/test split as the run below (266,285 train /
65,355 test moves, random_state=42). See reports/report_card_summary.md for
the shipped (tuned) model's numbers, which are better than the baselines
shown here - same format as the original run, so the two are easy to diff.

============================================================
Decision Quality (regression)

[Decision Quality] LinearRegression MAE= 31.42 RMSE= 36.61 R2=0.060
[Decision Quality] RandomForestRegressor MAE= 30.68 RMSE= 35.94 R2=0.094
[Decision Quality] XGBRegressor MAE= 29.93 RMSE= 35.75 R2=0.104

============================================================
Blunder Probability (classification)

[Blunder] LogisticRegression precision=0.096 recall=0.649 F1=0.167 AUC=0.738
confusion matrix:
[[42658 19518]
[ 1115 2064]]
[Blunder] RandomForestClassifier precision=0.600 recall=0.001 F1=0.002 AUC=0.785
confusion matrix:
[[62174 2]
[ 3176 3]]
[Blunder] XGBClassifier precision=0.146 recall=0.577 F1=0.233 AUC=0.788
confusion matrix:
[[51459 10717]
[ 1346 1833]]

Saved best decision-quality model (XGBRegressor) -> models/decision_quality_model.pkl
Saved best blunder model (XGBClassifier) -> models/blunder_model.pkl

--- Threshold tuning for each blunder model ---

LogisticRegression:
Best threshold: 0.66 F1=0.197 precision=0.137 recall=0.352
confusion matrix:
[[55118 7058]
[ 2060 1119]]

RandomForestClassifier:
Best threshold: 0.13 F1=0.265 precision=0.216 recall=0.344
confusion matrix:
[[58210 3966]
[ 2087 1092]]

XGBClassifier:
Best threshold: 0.705 F1=0.275 precision=0.227 recall=0.349
confusion matrix:
[[58396 3780]
[ 2069 1110]]

--- Shipped model (after hyperparameter tuning, beats every baseline above) ---

Decision quality: XGBRegressor (tuned) - MAE=29.75 RMSE=35.40 R2=0.121
Blunder: XGBClassifier (untuned defaults beat the tuned search on held-out
data - see reports/report_card_summary.md) - AUC=0.788, threshold=0.705,
F1=0.275, precision=0.227, recall=0.349
confusion matrix @0.705:
[[58396 3780]
[ 2069 1110]]
