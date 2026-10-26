# Step 15 — Dashboard (Streamlit)

**Status: ⏳ Do not start until Steps 12 and 14 exist.** The roadmap says this
explicitly: "this is not our immediate priority. We need the data and model working
first." A dashboard connected to a model that doesn't exist has nothing to display.

## What it will eventually show (for planning purposes only)

Per-player summary:
- Player, Rating
- Dynamic Decision Quality
- Blunder Probability
- Risk Level

Over-time / detail views:
- Evaluation over time
- Decision quality over time
- Remaining clock
- Blunder-risk timeline
- Accuracy trend
- Feature importance
- SHAP explanations (from Step 14 — the dashboard's job is to *display* these, not
  recompute them)

## What to call, and why (when you get here)

- `streamlit` (`pip install streamlit`, not yet in `requirements.txt` — add it when you
  actually start this step) for the app itself; `st.line_chart` /
  `st.plotly_chart` (you already have `plotly` installed) for the time-series views.
- Load your trained models with `pickle`/`joblib` rather than retraining inside the
  dashboard — the dashboard's job is to serve predictions, not train them.
- For the SHAP visuals, `shap.force_plot` and similar functions can render as
  matplotlib figures, which `st.pyplot()` can display directly.

## Resources

- Streamlit docs: https://docs.streamlit.io/
- Streamlit + SHAP integration is a common enough combo that searching
  "streamlit shap dashboard example" will turn up worked examples once you're here.
