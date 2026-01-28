"""
Fraud detection streaming dashboard (Streamlit).

Run from project root:
    streamlit run src/dashboards/streamlit_app.py

Data and model paths (override via env):
- Data: data/fraudTest.csv (or FRAUD_DATA_DIR)
- Model: models/model_pipeline.pkl, models/threshold.pkl or models/optimal_threshold.pkl
  or models/model_metadata.json (or FRAUD_MODELS_DIR)

Controls (sidebar):
- Threshold: slider 0.1–0.99. Used for high‑risk definition, cumulative metrics,
  and both confusion matrices. Default from saved threshold.
- Window size: 500 / 1000 / 2000 / 3000. Sliding-window size for metrics.
  Larger → more stable, fewer NaN; 2000 is a balanced default.
- History every: 50 or 100. Save metric history every N transactions.
- Min samples: 100 or 200. Minimum transactions before computing metrics.
- Stream speed: fast (no sleep) or real‑time (optional).
- Batch size: transactions to process per Run click.
- Run: advance simulation by one batch. Pause: stop advancing. Unpause: resume.
  Reset: clear history, re‑create tracker and scorer with current params.
"""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add project root to Python path for imports
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.dashboards.streamlit_helpers import (
    get_feature_lists,
    load_model_threshold_data,
)
from src.models.scoring_service import ScoringService
from src.monitoring.metrics import MetricsTracker


def _ensure_project_root():
    """Ensure cwd is project root so data/models paths resolve."""
    root = Path(__file__).resolve().parent.parent.parent
    import os
    if os.getcwd() != str(root):
        os.chdir(root)
    # Add project root to Python path for imports
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _init_session_state():
    if "initialized" in st.session_state:
        return
    _ensure_project_root()
    try:
        pipeline, default_threshold, segment, num_f, cat_f = load_model_threshold_data(
            use_segment=True, min_fraud=5, segment_size=2000
        )
    except Exception as e:
        st.session_state["load_error"] = str(e)
        st.session_state["initialized"] = True
        return

    st.session_state["pipeline"] = pipeline
    st.session_state["default_threshold"] = default_threshold
    st.session_state["segment"] = segment
    st.session_state["num_features"] = num_f
    st.session_state["cat_features"] = cat_f
    st.session_state["segment_records"] = segment.to_dict("records")

    st.session_state["transaction_history"] = []
    st.session_state["stream_index"] = 0
    st.session_state["run_start_time"] = None
    st.session_state["paused"] = False
    st.session_state["batch_size"] = 50

    _create_scorer_and_tracker(
        default_threshold,
        num_f,
        cat_f,
        window_size=2000,
        history_every=100,
        min_samples=200,
    )
    st.session_state["load_error"] = None
    st.session_state["initialized"] = True


def _create_scorer_and_tracker(
    threshold,
    num_f,
    cat_f,
    window_size=2000,
    history_every=100,
    min_samples=200,
):
    pipeline = st.session_state["pipeline"]
    scorer = ScoringService(
        pipeline=pipeline,
        threshold=threshold,
        numerical_features=num_f,
        categorical_features=cat_f,
    )
    tracker = MetricsTracker(
        window_size=window_size,
        threshold=threshold,
        history_every=history_every,
        min_samples=min_samples,
    )
    st.session_state["scorer"] = scorer
    st.session_state["metrics_tracker"] = tracker


def _run_batch(batch_size):
    records = st.session_state["segment_records"]
    idx = st.session_state["stream_index"]
    if idx >= len(records):
        return
    chunk = records[idx : idx + batch_size]
    scorer = st.session_state["scorer"]
    tracker = st.session_state["metrics_tracker"]
    history = st.session_state["transaction_history"]

    for txn in chunk:
        score = scorer.predict_proba(txn)
        label = int(txn.get("is_fraud", 0))
        tracker.update(txn, score, label)
        history.append({
            "timestamp": txn.get("trans_date_trans_time", datetime.utcnow()),
            "score": score,
            "label": label,
            "amount": float(txn.get("amt", 0) or 0),
            "category": str(txn.get("category", "unknown")),
        })

    st.session_state["stream_index"] = idx + len(chunk)
    if st.session_state["run_start_time"] is None:
        st.session_state["run_start_time"] = datetime.utcnow()


def _kpis(threshold):
    history = st.session_state.get("transaction_history", [])
    stats = st.session_state.get("scorer")
    stats = stats.get_stats() if stats else {}
    run_start = st.session_state.get("run_start_time")
    n = len(history)
    fraud = sum(h["label"] for h in history)
    high_risk = sum(1 for h in history if h["score"] > threshold)
    frate = (fraud / n * 100) if n else 0
    elapsed = 1.0
    if run_start and n:
        now = datetime.utcnow()
        delta = (now - run_start).total_seconds()
        elapsed = max(1.0, delta)
    throughput = n / elapsed if elapsed else 0
    avg_lat = stats.get("avg_latency_ms", 0) or 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total processed", n)
    c2.metric("Fraud count", fraud)
    c3.metric("Fraud rate %", f"{frate:.2f}")
    c4.metric("High‑risk count", high_risk)
    c5.metric("Avg latency ms", f"{avg_lat:.2f}")
    c6.metric("Throughput txns/s", f"{throughput:.1f}")


def _tab_stream_high_risk(threshold, last_n=1000):
    history = st.session_state.get("transaction_history", [])
    if not history:
        st.info("No transactions yet. Use Run in the sidebar.")
        return
    df = pd.DataFrame(history[-last_n:])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["score"],
            mode="lines",
            name="Score",
            line=dict(color="rgba(100,149,237,0.5)", width=1),
        )
    )
    high = df[df["score"] > threshold]
    if not high.empty:
        fig.add_trace(
            go.Scatter(
                x=high["timestamp"],
                y=high["score"],
                mode="markers",
                name=f"High‑risk (>{threshold:.2f})",
                marker=dict(symbol="triangle-up", size=8, color="orange"),
            )
        )
    fraud_df = df[df["label"] == 1]
    if not fraud_df.empty:
        fig.add_trace(
            go.Scatter(
                x=fraud_df["timestamp"],
                y=fraud_df["score"],
                mode="markers",
                name="Confirmed fraud",
                marker=dict(symbol="x", size=12, color="red"),
            )
        )
    fig.add_hline(y=threshold, line_dash="dash", line_color="orange")
    fig.update_layout(
        title="Scores over time",
        xaxis_title="Time",
        yaxis_title="Fraud score",
        yaxis_range=[0, 1.05],
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    high_risk_all = [h for h in history if h["score"] > threshold]
    st.subheader("High‑risk transactions")
    if not high_risk_all:
        st.write("None yet (score > threshold).")
        return
    hr_df = pd.DataFrame(sorted(high_risk_all, key=lambda x: -x["score"]))
    hr_df["ts"] = pd.to_datetime(hr_df["timestamp"]).dt.strftime("%H:%M:%S")
    hr_df["fraud"] = hr_df["label"].map({0: "No", 1: "Fraud"})
    st.dataframe(
        hr_df[["ts", "score", "amount", "category", "fraud"]].rename(
            columns={"ts": "time", "fraud": "label"}
        ),
        use_container_width=True,
    )


def _tab_metrics(threshold):
    tracker = st.session_state.get("metrics_tracker")
    history = st.session_state.get("transaction_history", [])
    mdf = tracker.get_metric_history_df() if tracker else pd.DataFrame()

    st.subheader("Time‑series metrics")
    if mdf.empty or "timestamp" not in mdf.columns:
        st.write("Insufficient data for metrics.")
    else:
        recent = mdf.tail(20)
        fig = go.Figure()
        for col, name in [("roc_auc", "ROC‑AUC"), ("pr_auc", "PR‑AUC")]:
            if col not in recent.columns:
                continue
            v = recent[col].dropna()
            if v.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=recent.loc[v.index, "timestamp"],
                    y=v,
                    mode="lines+markers",
                    name=name,
                )
            )
        if fig.data:
            fig.update_layout(
                title="ROC‑AUC / PR‑AUC (last 20 points)",
                xaxis_title="Time",
                yaxis_title="Score",
                yaxis_range=[0, 1.05],
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No valid ROC‑AUC / PR‑AUC yet.")

        fig2 = go.Figure()
        for col in ("precision", "recall", "f1"):
            if col not in recent.columns:
                continue
            v = recent[col].dropna()
            if v.empty:
                continue
            fig2.add_trace(
                go.Scatter(
                    x=recent.loc[v.index, "timestamp"],
                    y=v,
                    mode="lines+markers",
                    name=col.title(),
                )
            )
        if fig2.data:
            fig2.update_layout(
                title="Precision / Recall / F1 (last 20 points)",
                xaxis_title="Time",
                yaxis_range=[0, 1.05],
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        if "fraud_rate" in recent.columns:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["fraud_rate"],
                    mode="lines+markers",
                    name="Fraud rate",
                )
            )
            fig3.update_layout(
                title="Fraud rate over time",
                xaxis_title="Time",
                height=250,
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Cumulative metrics (all transactions)")
    if not history:
        st.write("No transactions yet.")
        return

    mdf_export = tracker.get_metric_history_df() if tracker else pd.DataFrame()
    if not mdf_export.empty:
        n = len(history)
        fcount = int(sum(h["label"] for h in history))
        ws = getattr(tracker, "window_size", "")
        he = getattr(tracker, "history_every", "")
        summary_line = f"# total_processed={n}, fraud_count={fcount}, window_size={ws}, history_every={he}\n"
        export_csv = summary_line + mdf_export.to_csv(index=False)
        st.download_button(
            "Download metrics (CSV)",
            data=export_csv,
            file_name="fraud_metrics.csv",
            mime="text/csv",
        )

    df = pd.DataFrame(history)
    scores = np.array([h["score"] for h in history])
    labels = np.array([h["label"] for h in history])
    preds = (scores >= threshold).astype(int)
    n = len(labels)
    fraud_n = int(labels.sum())
    high_risk_n = int((scores > threshold).sum())

    cum = {}
    if len(np.unique(labels)) >= 2:
        cum["roc_auc"] = roc_auc_score(labels, scores)
        cum["pr_auc"] = average_precision_score(labels, scores)
    else:
        cum["roc_auc"] = np.nan
        cum["pr_auc"] = np.nan
    cum["precision"] = precision_score(labels, preds, zero_division=0)
    cum["recall"] = recall_score(labels, preds, zero_division=0)
    cum["f1"] = f1_score(labels, preds, zero_division=0)
    cum["accuracy"] = accuracy_score(labels, preds)

    txt = (
        f"ROC‑AUC: {cum['roc_auc']:.4f}\n"
        f"PR‑AUC: {cum['pr_auc']:.4f}\n"
        f"Precision: {cum['precision']:.4f}\n"
        f"Recall: {cum['recall']:.4f}\n"
        f"F1: {cum['f1']:.4f}\n"
        f"Accuracy: {cum['accuracy']:.4f}\n"
        f"Fraud: {fraud_n} ({fraud_n / n * 100:.2f}%)\n"
        f"High‑risk: {high_risk_n}"
    )
    st.text(txt)


def _tab_confusion_matrices(threshold):
    tracker = st.session_state.get("metrics_tracker")
    history = st.session_state.get("transaction_history", [])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current window")
        if not tracker:
            st.write("Insufficient data.")
        else:
            scores, labels, _ = tracker.get_current_window()
            if len(scores) < 50:
                st.write("Insufficient data.")
            else:
                preds = (np.array(scores) >= threshold).astype(int)
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Pred 0", "Pred 1"],
                        y=["Actual 0", "Actual 1"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                    )
                )
                fig.update_layout(height=280, title="Window CM")
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cumulative")
        if not history:
            st.write("No transactions yet.")
        else:
            scores = np.array([h["score"] for h in history])
            labels = np.array([h["label"] for h in history])
            preds = (scores >= threshold).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred 0", "Pred 1"],
                    y=["Actual 0", "Actual 1"],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                )
            )
            fig.update_layout(height=280, title="Cumulative CM")
            st.plotly_chart(fig, use_container_width=True)


def _tab_operational():
    scorer = st.session_state.get("scorer")
    stats = scorer.get_stats() if scorer else {}
    if not stats:
        st.write("No scoring stats yet.")
        return
    st.metric("Total predictions", stats.get("total_predictions", 0))
    st.metric("Avg latency ms", f"{stats.get('avg_latency_ms', 0):.2f}")
    st.metric("P95 latency ms", f"{stats.get('p95_latency_ms', 0):.2f}")
    st.metric("P99 latency ms", f"{stats.get('p99_latency_ms', 0):.2f}")


def _tab_transaction_explorer(threshold):
    history = st.session_state.get("transaction_history", [])
    if not history:
        st.info("No transactions yet.")
        return
    filter_type = st.radio(
        "Filter",
        ["All", "Fraud only", "High‑risk only"],
        horizontal=True,
    )
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)
    filtered = history
    if filter_type == "Fraud only":
        filtered = [h for h in history if h["label"] == 1]
    elif filter_type == "High‑risk only":
        filtered = [h for h in history if h["score"] > threshold]
    filtered = [h for h in filtered if h["score"] >= min_score]
    df = pd.DataFrame(filtered)
    if df.empty:
        st.write("No transactions match filters.")
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df[["timestamp", "score", "amount", "category", "label"]], use_container_width=True)


def main():
    st.set_page_config(page_title="Fraud streaming dashboard", layout="wide")
    _init_session_state()

    if st.session_state.get("load_error"):
        st.error(f"Load failed: {st.session_state['load_error']}")
        st.info("Ensure data/fraudTest.csv and models/ (model_pipeline.pkl, threshold) exist.")
        return

    default_threshold = st.session_state["default_threshold"]
    num_f = st.session_state["num_features"]
    cat_f = st.session_state["cat_features"]

    with st.sidebar:
        st.header("Controls")
        threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=0.99,
            value=float(default_threshold),
            step=0.01,
            help="Used for high‑risk, cumulative metrics, and confusion matrices.",
        )
        window_size = st.selectbox(
            "Window size",
            options=[500, 1000, 2000, 3000],
            index=2,
            help="Sliding window for metrics.",
        )
        history_every = st.selectbox("History every", options=[50, 100], index=1)
        min_samples = st.selectbox("Min samples", options=[100, 200], index=1)
        stream_speed = st.radio("Stream speed", ["fast", "real-time"], index=0)
        batch_size = st.number_input("Batch size", min_value=1, max_value=200, value=50)

        st.caption(
            "Window size 2000 balances fraud coverage and stability. "
            "Larger windows: more stable metrics, fewer NaN."
        )

        col1, col2 = st.columns(2)
        with col1:
            run = st.button("Run", use_container_width=True)
            pause = st.button("Pause", use_container_width=True)
        with col2:
            unpause = st.button("Unpause", use_container_width=True)
            reset = st.button("Reset", use_container_width=True)

    if pause:
        st.session_state["paused"] = True
        st.rerun()
    if unpause:
        st.session_state["paused"] = False
        st.rerun()
    if reset:
        st.session_state["transaction_history"] = []
        st.session_state["stream_index"] = 0
        st.session_state["run_start_time"] = None
        st.session_state["paused"] = False
        _create_scorer_and_tracker(
            default_threshold,
            num_f,
            cat_f,
            window_size=window_size,
            history_every=history_every,
            min_samples=min_samples,
        )
        st.rerun()
    if run and not st.session_state.get("paused", False):
        _run_batch(batch_size)
        st.rerun()

    _kpis(threshold)

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Stream & high‑risk",
            "Metrics",
            "Confusion matrices",
            "Operational",
            "Transaction explorer",
        ]
    )
    with t1:
        _tab_stream_high_risk(threshold)
    with t2:
        _tab_metrics(threshold)
    with t3:
        _tab_confusion_matrices(threshold)
    with t4:
        _tab_operational()
    with t5:
        _tab_transaction_explorer(threshold)


if __name__ == "__main__":
    main()
