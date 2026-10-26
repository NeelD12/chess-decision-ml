@echo off
REM ============================================================
REM  Chess Decision ML - improvement pipeline
REM
REM  Hyperparameter-tunes every candidate model, scores all of them
REM  (baselines + tuned) on the same held-out test set, picks the best
REM  regressor and the best classifier by TEST performance (not just CV
REM  score), promotes them to models/decision_quality_model.pkl and
REM  models/blunder_model.pkl, tunes the blunder decision threshold and
REM  saves it to models/blunder_threshold.json, then regenerates
REM  reports/ against the promoted models.
REM
REM  Requires data/processed/ to already exist - run run_pipeline.bat
REM  first if it doesn't.
REM
REM  Usage:
REM    improve_models.bat
REM ============================================================

setlocal
cd /d "%~dp0"

if not exist logs mkdir logs
for /f "tokens=2 delims==" %%t in ('wmic os get localdatetime /value 2^>nul') do set "dt=%%t"
if defined dt (
    set "LOGFILE=logs\improve_%dt:~0,8%_%dt:~8,6%.log"
) else (
    set "LOGFILE=logs\improve_latest.log"
)
echo Logging to %LOGFILE%
echo.

powershell -NoProfile -Command "& venv\Scripts\python.exe src\models\improve_pipeline.py 2>&1 | Tee-Object -FilePath '%LOGFILE%' -Append"
if errorlevel 1 goto :error

echo.
echo ================================================
echo  Improvement pipeline complete. Full log: %LOGFILE%
echo    New scores: reports\report_card_summary.md
echo    Explore with: venv\Scripts\streamlit run dashboard\app.py
echo ================================================
pause
exit /b 0

:error
echo.
echo Pipeline stopped early - scroll up (or check %LOGFILE%) to see what failed.
pause
exit /b 1
