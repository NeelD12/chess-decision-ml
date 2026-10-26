| model | task | MAE | RMSE | R2 | precision | recall | f1 | auc |
|---|---|---|---|---|---|---|---|---|
| LinearRegression | decision_quality | 31.337 | 36.570 | 0.062 | nan | nan | nan | nan |
| RandomForestRegressor | decision_quality | 31.709 | 36.685 | 0.056 | nan | nan | nan | nan |
| XGBRegressor | decision_quality | 32.365 | 39.034 | -0.069 | nan | nan | nan | nan |
| LogisticRegression | blunder | nan | nan | nan | 0.120 | 0.657 | 0.202 | 0.777 |
| RandomForestClassifier | blunder | nan | nan | nan | 0.000 | 0.000 | 0.000 | 0.781 |
| XGBClassifier | blunder | nan | nan | nan | 0.205 | 0.132 | 0.161 | 0.740 |