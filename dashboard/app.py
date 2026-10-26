"""
Step 15 - Results Dashboard

Two different things happen in this file, on purpose:
  - The "Decision Quality" / "Blunder Probability" tabs retrain three
    candidate models each (cached per session) to compare them fairly -
    a baseline comparison needs all of them trained the same way.
  - The "Try a Prediction" tab does real inference: it loads whatever
    model training.py / tuning.py already saved to models/*.pkl and
    predicts with it directly - no retraining, the way a deployed model
    would actually be used (see src/models/prediction.py).

Run with:
    venv\\Scripts\\streamlit run dashboard\\app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
from training import load_data, make_group_split, train_decision_quality_models, train_blunder_models  # noqa: E402
from prediction import load_models, predict, analyze_uploaded_pgn, load_blunder_threshold  # noqa: E402
from evaluate_saved_models import evaluate_regression, evaluate_classification  # noqa: E402

st.set_page_config(page_title="Chess Decision Quality & Blunder Prediction", layout="wide")


@st.cache_data(show_spinner="Training baseline models (this only happens once per session)...")
def get_results():
    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]

    quality_results = train_decision_quality_models(X_train, X_test, y1_train, y1_test)
    blunder_results = train_blunder_models(X_train, X_test, y2_train, y2_test)

    n_games = game_id.nunique()
    return X, y1, y2, X_test, y1_test, y2_test, quality_results, blunder_results, n_games


@st.cache_resource(show_spinner="Loading saved models...")
def get_saved_models():
    # cache_resource (not cache_data) - these are live sklearn/xgboost model
    # objects, not serializable data, so they need the "keep the same Python
    # object across reruns" cache, not the "hash and store" one.
    return load_models()


st.title("Chess Decision Quality & Blunder Prediction")
st.caption(
    "Explainable ML on human decision quality under time pressure - "
    "baseline model results (Step 12), not yet the final tuned models."
)

X, y1, y2, X_test, y1_test, y2_test, quality_results, blunder_results, n_games = get_results()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Games", f"{n_games:,}")
col2.metric("Moves analyzed", f"{len(X):,}")
col3.metric("Blunder rate", f"{y2.mean() * 100:.1f}%")
col4.metric("Test set size", f"{len(X_test):,} moves")

st.info(
    "These are honest baseline results, not a polished final model - see the "
    "paper's methodology section for why weak/modest numbers here are expected "
    "and what a suspiciously perfect score would actually mean (a leakage bug, "
    "not a win - this project found and fixed exactly that once already).",
    icon="ℹ️",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Decision Quality (Regression)", "Blunder Probability (Classification)",
    "Try a Prediction", "Upload a Game", "Report Card (Saved Models)",
])

# ---------------------------------------------------------------------------
# Decision Quality
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Model comparison")
    quality_metrics_df = pd.DataFrame(
        {name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]} for name, r in quality_results.items()}
    ).T
    st.dataframe(quality_metrics_df.style.format("{:.3f}"), use_container_width=True)

    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["MAE", "RMSE", "R2"]):
        fig = px.bar(quality_metrics_df.reset_index(), x="index", y=metric, title=metric,
                     labels={"index": "model"})
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("Inspect a model")
    quality_model_choice = st.selectbox("Model", list(quality_results.keys()), key="quality_model")
    q_model = quality_results[quality_model_choice]["model"]
    q_preds = q_model.predict(X_test)

    left, right = st.columns(2)
    with left:
        scatter_df = pd.DataFrame({"Actual": y1_test.values, "Predicted": q_preds})
        fig = px.scatter(scatter_df, x="Actual", y="Predicted", opacity=0.3,
                          title=f"{quality_model_choice}: Actual vs Predicted")
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        residuals = y1_test.values - q_preds
        fig = px.scatter(x=q_preds, y=residuals, opacity=0.3,
                          labels={"x": "Predicted", "y": "Residual (actual - predicted)"},
                          title=f"{quality_model_choice}: Residuals")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    if hasattr(q_model, "feature_importances_"):
        importances = pd.Series(q_model.feature_importances_, index=X.columns).sort_values().tail(15)
        fig = px.bar(importances, orientation="h", title=f"{quality_model_choice}: Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Feature importance shown for tree-based models only (Random Forest / XGBoost).")

# ---------------------------------------------------------------------------
# Blunder Probability
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Model comparison")
    blunder_metrics_df = pd.DataFrame(
        {name: {"precision": r["precision"], "recall": r["recall"], "F1": r["f1"], "AUC": r["auc"]}
         for name, r in blunder_results.items()}
    ).T
    st.dataframe(blunder_metrics_df.style.format("{:.3f}"), use_container_width=True)

    metric_cols = st.columns(4)
    for col, metric in zip(metric_cols, ["precision", "recall", "F1", "AUC"]):
        fig = px.bar(blunder_metrics_df.reset_index(), x="index", y=metric, title=metric,
                     labels={"index": "model"}, range_y=[0, 1])
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC and Precision-Recall curves (all models)")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, r in blunder_results.items():
            fpr, tpr, _ = roc_curve(y2_test, r["model"].predict_proba(X_test)[:, 1])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                  name="Random guessing"))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        for name, r in blunder_results.items():
            precision, recall, _ = precision_recall_curve(y2_test, r["model"].predict_proba(X_test)[:, 1])
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
        base_rate = y2_test.mean()
        fig.add_hline(y=base_rate, line_dash="dash", line_color="gray",
                       annotation_text=f"Base rate ({base_rate:.2f})")
        fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inspect a model")
    blunder_model_choice = st.selectbox("Model", list(blunder_results.keys()), key="blunder_model")
    b_model = blunder_results[blunder_model_choice]["model"]
    b_preds = b_model.predict(X_test)

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y2_test, b_preds)
        fig = px.imshow(cm, text_auto=True, x=["No Blunder", "Blunder"], y=["No Blunder", "Blunder"],
                         labels=dict(x="Predicted", y="Actual"),
                         title=f"{blunder_model_choice}: Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if hasattr(b_model, "feature_importances_"):
            importances = pd.Series(b_model.feature_importances_, index=X.columns).sort_values().tail(15)
            fig = px.bar(importances, orientation="h", title=f"{blunder_model_choice}: Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Feature importance shown for tree-based models only (Random Forest / XGBoost).")

# ---------------------------------------------------------------------------
# Try a Prediction - real inference against a SAVED model, not a retrain.
# Everything above this point trains fresh candidate models to compare them
# fairly (that's what a baseline comparison needs) - this tab is different:
# it loads whatever model training.py / tuning.py already saved to models/*.pkl
# and predicts with it directly, the way an actual deployed model would be used.
# ---------------------------------------------------------------------------
def render_try_prediction_tab():
    st.subheader("Predict from a saved model (no retraining)")
    try:
        quality_model, blunder_model, quality_path, blunder_path = get_saved_models()
    except FileNotFoundError as e:
        st.error(str(e))
        return  # not st.stop() - that would halt the WHOLE script, killing tab4 too

    st.caption(
        f"Decision-quality model: `{quality_path.relative_to(PROJECT_ROOT)}`  |  "
        f"Blunder model: `{blunder_path.relative_to(PROJECT_ROOT)}`"
    )

    moves_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "moves_features.parquet")

    if "predict_row" not in st.session_state:
        st.session_state.predict_row = 0

    def _pick_random_row():
        # FIXED: session_state for a widget's key can only be set in a
        # callback (which runs BEFORE the widget is re-instantiated), never
        # directly inside the script body after that widget already exists -
        # doing it inline threw "cannot be modified after the widget ... is
        # instantiated" here.
        st.session_state.predict_row = int(np.random.randint(0, len(X)))

    pick_col, button_col = st.columns([4, 1])
    with pick_col:
        row_idx = st.slider("Pick a move (row index)", 0, len(X) - 1, key="predict_row")
    with button_col:
        st.write("")  # vertical alignment spacer
        st.button("🎲 Random move", on_click=_pick_random_row)

    context = moves_df.iloc[row_idx]
    X_row = X.iloc[[row_idx]]
    result = predict(quality_model, blunder_model, X_row)

    left, right = st.columns(2)
    with left:
        st.markdown("**Position context**")
        st.write(f"Game: `{context['game_id']}`, move {int(context['move_number'])} ({context['player']})")
        st.write(f"Played: `{context['played_move']}`")
        st.code(context["fen_before"], language=None)

        st.markdown("**Actual (ground truth)**")
        st.write(f"Decision quality: **{y1.iloc[row_idx]:.1f} / 100**")
        st.write(f"Was a blunder: **{'Yes' if y2.iloc[row_idx] else 'No'}**")

    with right:
        st.markdown("**Model prediction**")
        st.metric("Predicted decision quality", f"{result['decision_quality']:.1f} / 100")
        st.metric("Predicted blunder probability", f"{result['blunder_probability']:.1%}")
        st.metric("Blunder flag", "Yes" if result["blunder_flag"] else "No")
        st.caption(
            f"Flag uses threshold {result['blunder_threshold_used']:.2f} - the F1-optimal "
            f"cutoff from `models/blunder_threshold.json` if `improve_pipeline.py` has been "
            f"run, otherwise the sklearn default of 0.5 (not calibrated for how rare "
            f"blunders actually are - see docs/step12_model_training.md)."
        )


with tab3:
    render_try_prediction_tab()


# ---------------------------------------------------------------------------
# Upload a Game - real inference on BRAND NEW data, not a row from the
# training set. Runs the exact same pipeline the training data went through
# (parse -> Stockfish analysis -> feature engineering -> predict), live, on
# whatever PGN is uploaded. See src/models/prediction.analyze_uploaded_pgn().
# ---------------------------------------------------------------------------
def render_upload_tab():
    st.subheader("Upload your own game")
    st.caption(
        "Upload a PGN for one game, with player ratings (WhiteElo/BlackElo) and "
        "clock annotations (%clk) - e.g. export a game from Lichess with clock "
        "times included. This runs live Stockfish analysis on the whole game, "
        "so it takes a few seconds, not a few hours."
    )

    try:
        quality_model, blunder_model, quality_path, blunder_path = get_saved_models()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    uploaded = st.file_uploader("PGN file", type=["pgn"], key="pgn_upload")
    if uploaded is None:
        return

    pgn_text = uploaded.read().decode("utf-8", errors="replace")

    try:
        with st.spinner("Running Stockfish analysis on this game..."):
            result_df = analyze_uploaded_pgn(pgn_text, X.columns)
    except ValueError as e:
        st.error(str(e))
        return

    threshold = load_blunder_threshold()
    st.success(f"Analyzed {len(result_df)} moves. Blunder flag uses threshold {threshold:.2f}.")

    st.dataframe(
        result_df.style.format({
            "actual_decision_quality": "{:.0f}",
            "predicted_decision_quality": "{:.1f}",
            "predicted_blunder_probability": "{:.1%}",
        }),
        use_container_width=True,
        height=350,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df["move_number"], y=result_df["predicted_blunder_probability"],
        mode="lines+markers", name="Predicted blunder probability",
    ))
    fig.add_hline(y=threshold, line_dash="dot", line_color="orange",
                  annotation_text=f"flag threshold ({threshold:.2f})")
    flagged_moves = result_df[result_df["predicted_blunder_flag"]]
    fig.add_trace(go.Scatter(
        x=flagged_moves["move_number"], y=flagged_moves["predicted_blunder_probability"],
        mode="markers", marker=dict(size=10, symbol="circle-open", color="orange"),
        name="Flagged (over threshold)",
    ))
    blunder_moves = result_df[result_df["actual_blunder"]]
    fig.add_trace(go.Scatter(
        x=blunder_moves["move_number"], y=blunder_moves["predicted_blunder_probability"],
        mode="markers", marker=dict(size=12, symbol="x", color="red"),
        name="Actual blunder",
    ))
    fig.update_layout(
        title="Predicted blunder risk across the game",
        xaxis_title="Move number", yaxis_title="Predicted blunder probability",
    )
    st.plotly_chart(fig, use_container_width=True)


with tab4:
    render_upload_tab()


# ---------------------------------------------------------------------------
# Report Card - the ONE tab meant to be shared with someone outside the
# project (a mentor, a co-author) to "just get the scores." Unlike tab1/tab2,
# it never retrains anything - it scores exactly the two models saved to
# models/*.pkl, the ones that would actually ship, using the same evaluation
# code evaluate_saved_models.py uses from the command line (so the numbers
# here and the numbers in reports/report_card_summary.md always agree).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Scoring the saved models (no retraining)...")
def get_report_card():
    quality_model, blunder_model, quality_path, blunder_path = load_models()
    X, y1, y2, game_id = load_data()
    train_idx, test_idx = make_group_split(X, game_id)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y1_train, y1_test = y1.iloc[train_idx], y1.iloc[test_idx]
    y2_train, y2_test = y2.iloc[train_idx], y2.iloc[test_idx]

    q_result = evaluate_regression(quality_model, X_test, y1_test, y1_train)
    b_result = evaluate_classification(blunder_model, X_test, y2_test)

    meta = {
        "quality_path": quality_path, "blunder_path": blunder_path,
        "n_moves": len(X), "n_games": game_id.nunique(),
        "n_test_moves": len(X_test), "n_test_games": game_id.iloc[test_idx].nunique(),
        "n_train_rows": len(X_train),
    }
    return q_result, b_result, y1_test, y2_test, meta


def render_report_card_tab():
    st.subheader("Report card: the saved models, scored honestly")
    st.caption(
        "This is the one tab meant to be shared outside the team. It never retrains "
        "anything - it loads models/*.pkl exactly as they are and scores them against "
        "whatever is currently in data/processed/. Same numbers as running "
        "`python src/models/evaluate_saved_models.py` from the command line."
    )

    try:
        q_result, b_result, y1_test, y2_test, meta = get_report_card()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.markdown(
        f"**Decision-quality model:** `{meta['quality_path'].name}` ({q_result['model']})  \n"
        f"**Blunder model:** `{meta['blunder_path'].name}` ({b_result['model']})  \n"
        f"**Data on disk:** {meta['n_moves']:,} moves / {meta['n_games']:,} games "
        f"(test split: {meta['n_test_moves']:,} moves / {meta['n_test_games']:,} games)"
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("### Decision quality (regression)")
        st.metric("R2 (headline number - use this across runs, not MAE)", f"{q_result['R2']:.3f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{q_result['MAE']:.2f}")
        c2.metric("RMSE", f"{q_result['RMSE']:.2f}")
        c3.metric("Naive baseline MAE", f"{q_result['naive_baseline_MAE']:.2f}")
        st.caption(
            "MAE alone isn't comparable across differently-sized/sourced datasets - the "
            "test set's own variance changes it even with no real model improvement. R2 "
            "controls for that."
        )
        scatter_df = pd.DataFrame({"Actual": y1_test.values, "Predicted": q_result["preds"]})
        fig = px.scatter(scatter_df, x="Actual", y="Predicted", opacity=0.25,
                          title=f"{q_result['model']}: Actual vs Predicted")
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Blunder probability (classification)")
        st.metric("AUC (ranking ability - holds regardless of threshold)", f"{b_result['auc']:.3f}")
        st.caption(f"Base rate (actual blunders in test set): {b_result['base_rate']:.2%}")

        cmp_df = pd.DataFrame({
            "default (t=0.5)": {"precision": b_result["default"]["precision"],
                                 "recall": b_result["default"]["recall"],
                                 "F1": b_result["default"]["f1"]},
            f"tuned (t={b_result['tuned']['threshold']:.2f})": {
                "precision": b_result["tuned"]["precision"],
                "recall": b_result["tuned"]["recall"],
                "F1": b_result["tuned"]["f1"]},
        }).T
        st.dataframe(cmp_df.style.format("{:.3f}"), use_container_width=True)
        st.caption(
            "The default 0.5 cutoff is almost never right for a rare event like blunders "
            "(~5% base rate) - the tuned threshold is the F1-maximizing cutoff found by "
            "sweeping 0-1. Ship the tuned threshold, not the default, if this model goes "
            "into the website."
        )

        sweep = b_result["sweep"]
        fig = go.Figure()
        for col in ["precision", "recall", "f1"]:
            fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col], mode="lines", name=col))
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="default")
        fig.add_vline(x=b_result["tuned"]["threshold"], line_dash="dash", line_color="red",
                       annotation_text="tuned")
        fig.update_layout(title="Threshold sweep", xaxis_title="Threshold", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Confusion matrices")
    cm_left, cm_right = st.columns(2)
    for col, key, label in zip([cm_left, cm_right], ["default", "tuned"],
                                [f"Default (t=0.5)", f"Tuned (t={b_result['tuned']['threshold']:.2f})"]):
        cm = b_result[key]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["No Blunder", "Blunder"], y=["No Blunder", "Blunder"],
                         labels=dict(x="Predicted", y="Actual"),
                         title=f"{label} - F1={b_result[key]['f1']:.3f}", color_continuous_scale="Blues")
        col.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### What to put in the paper")
    st.markdown(
        f"- Decision quality: **MAE {q_result['MAE']:.2f}, RMSE {q_result['RMSE']:.2f}, "
        f"R2 {q_result['R2']:.3f}** ({q_result['model']}, n={meta['n_test_moves']:,} test moves)\n"
        f"- Blunder detection: **AUC {b_result['auc']:.3f}**, F1 {b_result['tuned']['f1']:.3f} "
        f"at the tuned threshold ({b_result['tuned']['threshold']:.2f}) vs. "
        f"{b_result['default']['f1']:.3f} at the conventional 0.5 cutoff "
        f"({b_result['model']}, base rate {b_result['base_rate']:.2%})\n"
        "- Report AUC/PR-curve as the headline blunder metric, not a single F1 at an "
        "arbitrary threshold - it's threshold-independent and standard for imbalanced "
        "classification."
    )


with tab5:
    render_report_card_tab()
