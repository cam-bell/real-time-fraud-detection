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

# Lazy import for v3 feature engineer (avoids Streamlit caching issues)
try:
    from src.features.feature_engineering import OnlineFeatureEngineerV3
except ImportError:
    OnlineFeatureEngineerV3 = None
from src.monitoring.metrics import MetricsTracker
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.alerts import AlertManager


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


def _available_model_versions(models_dir: Path):
    """
    Best-effort model version discovery.

    Kept local so Streamlit hot-reload / module caching can't break imports.
    """
    versions = ["v2"]
    if (models_dir / "fraud_detection_pipeline_v3.pkl").exists():
        versions.append("v3")
    return versions


def _features_dict_from_txn(txn, num_features):
    """Build numeric features dict for DriftDetector from transaction."""
    out = {}
    for k in num_features:
        v = txn.get(k, 0)
        try:
            out[k] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def _init_session_state():
    if "initialized" in st.session_state:
        return
    _ensure_project_root()
    try:
        model_version = "v2"
        result = load_model_threshold_data(
            use_segment=True, min_fraud=5, segment_size=2000, model_version=model_version
        )
        # Handle both old (5-tuple) and new (7-tuple) return signatures
        if len(result) == 5:
            pipeline, thresholds, segment, num_f, cat_f = result
            target_encodings = {}
            feature_meta = {}
        else:
            pipeline, thresholds, segment, num_f, cat_f, target_encodings, feature_meta = result
    except Exception as e:
        st.session_state["load_error"] = str(e)
        st.session_state["initialized"] = True
        return

    st.session_state["pipeline"] = pipeline
    # Backward-compat: older helper may return a single float threshold.
    if not isinstance(thresholds, dict):
        thresholds = {"cost": float(thresholds), "f1": float(thresholds)}

    st.session_state["model_version"] = model_version
    st.session_state["thresholds"] = thresholds
    st.session_state["preset_balanced_thresholds"] = {
        "cost": float(thresholds.get("cost", 0.5)),
        "f1": float(thresholds.get("f1", 0.5)),
    }
    st.session_state["default_threshold"] = float(thresholds.get("cost", 0.5))
    st.session_state["segment"] = segment
    st.session_state["num_features"] = num_f
    st.session_state["cat_features"] = cat_f
    st.session_state["target_encodings"] = target_encodings
    st.session_state["feature_meta"] = feature_meta
    st.session_state["segment_records"] = segment.to_dict("records")

    st.session_state["transaction_history"] = []
    st.session_state["stream_index"] = 0
    st.session_state["run_start_time"] = None
    st.session_state["paused"] = False
    st.session_state["batch_size"] = 50
    st.session_state["drift_enabled"] = False
    st.session_state["drift_check_every"] = 100
    st.session_state["last_drift_check_n"] = 0
    st.session_state["last_drift_status"] = None
    # Card history cache for v3 velocity features
    if "card_history_cache" not in st.session_state:
        from collections import defaultdict
        st.session_state["card_history_cache"] = defaultdict(list)

    _create_scorer_and_tracker(
        st.session_state["default_threshold"],
        num_f,
        cat_f,
        model_version=model_version,
        target_encodings=target_encodings,
        feature_meta=feature_meta,
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
    model_version="v2",
    target_encodings=None,
    feature_meta=None,
    window_size=2000,
    history_every=100,
    min_samples=200,
):
    pipeline = st.session_state["pipeline"]
    
    # Create v3-compatible feature engineer if needed
    feature_engineer = None
    if model_version == "v3":
        if OnlineFeatureEngineerV3 is None:
            raise ImportError(
                "OnlineFeatureEngineerV3 not available. "
                "Please restart Streamlit to reload the module."
            )
        cat_lists = st.session_state.get("feature_meta", {}).get("category_risk_lists", {})
        high = cat_lists.get("high_risk", ["shopping_net", "grocery_pos", "misc_net", "home"])
        medium = cat_lists.get("medium_risk", ["food_dining", "kids_pets", "health_fitness"])
        card_cache = st.session_state.get("card_history_cache")
        all_model_features = st.session_state.get("feature_meta", {}).get("all_model_features", num_f + cat_f)
        feature_engineer = OnlineFeatureEngineerV3(
            numerical_features=num_f,
            categorical_features=cat_f,
            target_encodings=target_encodings or {},
            high_risk_categories=high,
            medium_risk_categories=medium,
            card_history_cache=card_cache,
            all_model_features=all_model_features,
        )
    
    scorer = ScoringService(
        pipeline=pipeline,
        threshold=threshold,
        numerical_features=num_f if feature_engineer is None else None,
        categorical_features=cat_f if feature_engineer is None else None,
        feature_engineer=feature_engineer,
    )
    tracker = MetricsTracker(
        window_size=window_size,
        threshold=threshold,
        history_every=history_every,
        min_samples=min_samples,
    )
    st.session_state["scorer"] = scorer
    st.session_state["metrics_tracker"] = tracker


def _create_drift_and_alerts(high_risk_threshold=0.9):
    """Create DriftDetector and AlertManager; store in session_state."""
    detector = DriftDetector(
        reference_window=500,
        detection_window=500,
        ks_alpha=0.01,
        ks_stat_threshold=0.1,
        performance_drop_threshold=0.05,
        cooldown_after_retrain=500,
    )
    alerts = AlertManager(
        high_risk_threshold=high_risk_threshold,
        performance_drop_threshold=0.05,
    )
    st.session_state["drift_detector"] = detector
    st.session_state["alert_manager"] = alerts


def _run_batch(batch_size):
    records = st.session_state["segment_records"]
    idx = st.session_state["stream_index"]
    if idx >= len(records):
        return
    chunk = records[idx : idx + batch_size]
    scorer = st.session_state["scorer"]
    tracker = st.session_state["metrics_tracker"]
    history = st.session_state["transaction_history"]
    drift_enabled = st.session_state.get("drift_enabled", False)
    drift_check_every = st.session_state.get("drift_check_every", 100)
    num_f = st.session_state.get("num_features", [])
    ref_window = 500
    min_for_drift_check = ref_window + 500  # ref + detection full

    if drift_enabled:
        if "drift_detector" not in st.session_state or st.session_state["drift_detector"] is None:
            thr = st.session_state.get("thresholds", {}) or {}
            cost_thr = float(thr.get("cost") or st.session_state.get("default_threshold") or 0.9)
            _create_drift_and_alerts(high_risk_threshold=cost_thr)
        detector = st.session_state["drift_detector"]
        alerts = st.session_state["alert_manager"]
    last_drift_check_n = st.session_state.get("last_drift_check_n", 0)

    for txn in chunk:
        score = scorer.predict_proba(txn)
        label = int(txn.get("is_fraud", 0))
        tracker.update(txn, score, label)
        thr = st.session_state.get("thresholds", {}) or {}
        cost_thr = float(thr.get("cost") or st.session_state.get("default_threshold") or 0.5)
        f1_thr = float(thr.get("f1") or cost_thr)
        review_thr = min(cost_thr, f1_thr)
        block_thr = max(cost_thr, f1_thr)
        if score >= block_thr:
            decision = "block"
        elif score >= review_thr:
            decision = "review"
        else:
            decision = "approve"
        history.append({
            "timestamp": txn.get("trans_date_trans_time", datetime.utcnow()),
            "score": score,
            "label": label,
            "amount": float(txn.get("amt", 0) or 0),
            "category": str(txn.get("category", "unknown")),
            "decision": decision,
        })
        n = len(history)

        if drift_enabled and num_f:
            feats = _features_dict_from_txn(txn, num_f)
            if n <= ref_window:
                detector.update_reference(score, feats)
            else:
                detector.update_detection(score, feats)
            alerts.check_high_risk_transaction(txn, score)

            if n >= min_for_drift_check and (n - last_drift_check_n) >= drift_check_every:
                latest = tracker.get_latest_metrics() or {}
                current_roc = latest.get("roc_auc")
                try:
                    _roc_ok = current_roc is not None and np.isfinite(float(current_roc))
                except (TypeError, ValueError):
                    _roc_ok = False
                if _roc_ok:
                    if detector.baseline_roc_auc is None:
                        detector.set_baseline_performance(current_roc)
                        alerts.set_baseline_performance(current_roc)
                    drift_status = detector.check_drift(current_roc_auc=current_roc, transaction_count=n)
                    st.session_state["last_drift_status"] = drift_status
                    alerts.check_performance_degradation(current_roc)
                    alerts.check_drift_alert(drift_status)
                last_drift_check_n = n

    st.session_state["last_drift_check_n"] = last_drift_check_n
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
    block_n = sum(1 for h in history if h.get("decision") == "block")
    review_n = sum(1 for h in history if h.get("decision") == "review")
    approved_n = sum(1 for h in history if h.get("decision") == "approve")
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
    if st.session_state.get("dual_threshold", False):
        c4.metric("Review queue", review_n)
        c5.metric("Auto blocks", block_n)
        c6.metric("Approved", approved_n)
        st.caption(
            f"Throughput: {throughput:.1f} txns/s | Avg latency: {avg_lat:.2f} ms — "
            "**Review queue**: transactions sent to manual review (score between review and block thresholds)."
        )
    else:
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
    dual = st.session_state.get("dual_threshold", False)
    review_thr, block_thr = threshold, threshold
    if dual:
        thr = st.session_state.get("thresholds", {}) or {}
        cost_thr = float(thr.get("cost", threshold))
        f1_thr = float(thr.get("f1", threshold))
        review_thr = min(cost_thr, f1_thr)
        block_thr = max(cost_thr, f1_thr)
        fig.add_hline(y=block_thr, line_dash="dot", line_color="red")
        # Shaded review zone
        fig.add_hrect(
            y0=review_thr, y1=block_thr,
            fillcolor="orange", opacity=0.15, line_width=0,
            annotation_text="Review zone", annotation_position="top left",
        )
    fig.update_layout(
        title="Scores over time — Approve | Review | Block" if dual else "Scores over time",
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
    hr_df["fraud"] = hr_df["label"].replace({0: "No", 1: "Fraud"})
    display_df = hr_df.loc[:, ["ts", "score", "amount", "category", "fraud"]].rename(
        columns={"ts": "time", "fraud": "label"}
    )
    st.dataframe(
        display_df,
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

    # Export transaction history with decisions for audit
    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    hist_df = hist_df.rename(columns={"label": "is_fraud"})
    cols = ["timestamp", "score", "amount", "category", "is_fraud"]
    if "decision" in hist_df.columns:
        cols.append("decision")
    export_hist = hist_df[cols].to_csv(index=False)
    st.download_button(
        "Download history (with decisions)",
        data=export_hist,
        file_name="fraud_history_with_decisions.csv",
        mime="text/csv",
        key="download_history_decisions",
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
    cum["precision"] = precision_score(labels, preds, zero_division="warn")
    cum["recall"] = recall_score(labels, preds, zero_division="warn")
    cum["f1"] = f1_score(labels, preds, zero_division="warn")
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

    if st.session_state.get("dual_threshold", False):
        st.divider()
        thr = st.session_state.get("thresholds", {}) or {}
        cost_thr = float(thr.get("cost", threshold))
        f1_thr = float(thr.get("f1", threshold))
        block_thr = max(cost_thr, f1_thr)
        st.subheader(f"Auto-block confusion matrices (thr={block_thr:.2f})")
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Current window")
            if tracker:
                scores, labels, _ = tracker.get_current_window()
                if len(scores) >= 50:
                    preds = (np.array(scores) >= block_thr).astype(int)
                    cm = confusion_matrix(labels, preds, labels=[0, 1])
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=cm,
                            x=["Pred 0", "Pred 1"],
                            y=["Actual 0", "Actual 1"],
                            text=cm,
                            texttemplate="%{text}",
                            colorscale="Reds",
                        )
                    )
                    fig.update_layout(height=260, title="Window CM (block)")
                    st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.caption("Cumulative")
            if history:
                scores = np.array([h["score"] for h in history])
                labels = np.array([h["label"] for h in history])
                preds = (scores >= block_thr).astype(int)
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Pred 0", "Pred 1"],
                        y=["Actual 0", "Actual 1"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Reds",
                    )
                )
                fig.update_layout(height=260, title="Cumulative CM (block)")
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


def _drift_status_indicator():
    """Render green/yellow/red drift status badge."""
    drift_enabled = st.session_state.get("drift_enabled", False)
    last = st.session_state.get("last_drift_status")
    n = len(st.session_state.get("transaction_history", []))

    if not drift_enabled:
        st.caption("Drift: off")
        return
    if n < 1000:
        st.warning("Drift: warming up (≥1000 txns)")
        return
    if last is None:
        st.warning("Drift: no check yet")
        return
    if last.get("drift_detected"):
        st.error("Drift detected")
        return
    st.success("Drift: OK")


def _tab_drift_alerts(threshold):
    """Drift status, KS stats when drift detected, alert log."""
    drift_enabled = st.session_state.get("drift_enabled", False)
    if not drift_enabled:
        st.info("Enable **Drift monitoring** in the sidebar to use this tab.")
        return

    detector = st.session_state.get("drift_detector")
    alerts = st.session_state.get("alert_manager")
    last = st.session_state.get("last_drift_status")

    st.subheader("Drift status")
    if last is None:
        st.write("No drift check yet. Run the simulation until ≥1000 transactions, then checks run every N.")
    else:
        drift_ok = not last.get("drift_detected", False)
        if drift_ok:
            st.success("No drift")
        else:
            st.error("Drift detected")
        details = last.get("details", {})
        if details:
            if "score_ks_stat" in details:
                st.code(
                    f"Score KS stat: {details['score_ks_stat']:.4f}\n"
                    f"Score KS p-value: {details['score_ks_p']:.6f}"
                )
            if "performance_drop" in details:
                st.code(f"ROC-AUC drop: {details['performance_drop']:.4f}")
            if "feature_drifts" in details:
                st.write("Feature drift (KS):")
                for fd in details["feature_drifts"][:5]:
                    st.code(f"  {fd['feature']}: stat={fd['ks_stat']:.4f} p={fd['p_value']:.6f}")

    st.subheader("Alert log")
    if alerts is None:
        st.write("No alerts yet. Run the simulation with drift enabled to populate.")
        return
    recent = alerts.get_recent_alerts(20)
    if not recent:
        st.write("No alerts yet.")
        return
    rows = []
    for a in recent:
        ts = a.get("timestamp") or datetime.utcnow()
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        rows.append({
            "Time": ts_str,
            "Type": a.get("type", ""),
            "Severity": a.get("severity", ""),
            "Message": a.get("message", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _tab_review_queue(threshold):
    """List transactions in review queue (and optionally blocks) for dual-threshold workflow."""
    history = st.session_state.get("transaction_history", [])
    if not history:
        st.info("No transactions yet. Run the simulation with **Dual-threshold workflow** enabled.")
        return
    include_blocks = st.checkbox("Include auto-blocks", value=False, help="Show blocked transactions as well.")
    decisions = ["review", "block"] if include_blocks else ["review"]
    filtered = [h for h in history if h.get("decision") in decisions]
    if not filtered:
        st.info("No transactions in review queue." + (" Try enabling **Include auto-blocks**." if not include_blocks else ""))
        return
    df = pd.DataFrame(filtered)
    df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
    df["label"] = df["label"].replace({0: "No", 1: "Fraud"})
    st.dataframe(
        df[["time", "score", "amount", "category", "decision", "label"]],
        use_container_width=True,
        column_config={"decision": st.column_config.TextColumn("Decision")},
    )


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
        models_dir = Path("models")
        versions = _available_model_versions(models_dir=models_dir)
        model_version = st.selectbox("Model version", options=versions, index=versions.index(st.session_state.get("model_version", "v2")))
        dual_threshold = st.checkbox(
            "Dual-threshold workflow (review + block)",
            value=st.session_state.get("dual_threshold", model_version == "v3"),
            help="Two thresholds: lower = manual review queue; higher = auto-block. Approve < Review queue < Auto block.",
        )
        st.session_state["dual_threshold"] = bool(dual_threshold)

        if model_version != st.session_state.get("model_version"):
            result = load_model_threshold_data(
                use_segment=True, min_fraud=5, segment_size=2000, model_version=model_version
            )
            # Handle both old (5-tuple) and new (7-tuple) return signatures
            if len(result) == 5:
                pipeline, thresholds, segment, num_f, cat_f = result
                target_encodings = {}
                feature_meta = {}
            else:
                pipeline, thresholds, segment, num_f, cat_f, target_encodings, feature_meta = result
            
            if not isinstance(thresholds, dict):
                thresholds = {"cost": float(thresholds), "f1": float(thresholds)}
            st.session_state["pipeline"] = pipeline
            st.session_state["thresholds"] = thresholds
            st.session_state["preset_balanced_thresholds"] = {
                "cost": float(thresholds.get("cost", 0.5)),
                "f1": float(thresholds.get("f1", 0.5)),
            }
            st.session_state["default_threshold"] = float(thresholds.get("cost", 0.5))
            st.session_state["segment"] = segment
            st.session_state["num_features"] = num_f
            st.session_state["cat_features"] = cat_f
            st.session_state["target_encodings"] = target_encodings
            st.session_state["feature_meta"] = feature_meta
            st.session_state["segment_records"] = segment.to_dict("records")
            st.session_state["transaction_history"] = []
            st.session_state["stream_index"] = 0
            st.session_state["run_start_time"] = None
            st.session_state["model_version"] = model_version
            # Reset card history cache when switching models
            from collections import defaultdict
            st.session_state["card_history_cache"] = defaultdict(list)
            _create_scorer_and_tracker(
                st.session_state["default_threshold"],
                num_f,
                cat_f,
                model_version=model_version,
                target_encodings=target_encodings,
                feature_meta=feature_meta,
                window_size=2000,
                history_every=100,
                min_samples=200,
            )
            st.rerun()

        if dual_threshold:
            thr = st.session_state.get("thresholds", {}) or {}
            balanced = st.session_state.get("preset_balanced_thresholds") or thr
            cost_default = float(thr.get("cost", default_threshold))
            f1_default = float(thr.get("f1", max(cost_default, 0.9)))

            # Fixed preset values so Balanced/Conservative/Aggressive always revert correctly
            preset_options = ["Custom", "Balanced", "Conservative", "Aggressive"]
            preset_labels = {
                "Balanced": "Model defaults (cost + F1) ",
                "Conservative": "More review, fewer blocks",
                "Aggressive": "Fewer review, more blocks",
            }
            preset = st.selectbox(
                "Threshold preset",
                options=preset_options,
                format_func=lambda x: str(preset_labels.get(x, x)),
                help="Presets use fixed thresholds; Custom keeps current sliders.",
            )
            if preset != "Custom":
                if preset == "Balanced":
                    review_thr = float(balanced.get("cost", 0.83))
                    block_thr = float(balanced.get("f1", 0.98))
                elif preset == "Conservative":
                    review_thr, block_thr = 0.78, 0.99
                else:  # Aggressive
                    review_thr, block_thr = 0.83, 0.94
                st.session_state["thresholds"] = {"cost": review_thr, "f1": block_thr}
                if st.session_state.get("threshold_preset") != preset:
                    st.session_state["threshold_preset"] = preset
                    st.rerun()
            else:
                st.session_state["threshold_preset"] = "Custom"

            review_thr = st.slider(
                "Review threshold (lower)",
                min_value=0.1,
                max_value=0.99,
                value=float(min(cost_default, f1_default)),
                step=0.01,
                help="Scores ≥ this go to the manual review queue (not auto-blocked).",
            )
            block_thr = st.slider(
                "Auto-block threshold (higher)",
                min_value=0.1,
                max_value=0.99,
                value=float(max(cost_default, f1_default)),
                step=0.01,
                help="Scores ≥ this are auto-blocked; between review and block = review queue only.",
            )
            st.session_state["thresholds"] = {"cost": float(review_thr), "f1": float(block_thr)}
            default_threshold = float(review_thr)
            threshold = float(review_thr)
        else:
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

        st.divider()
        st.subheader("Drift & alerts")
        drift_enabled = st.checkbox(
            "Enable drift monitoring",
            value=st.session_state.get("drift_enabled", False),
            help="Use DriftDetector + AlertManager; trigger check every N transactions.",
        )
        st.session_state["drift_enabled"] = drift_enabled
        if drift_enabled:
            drift_check_every = st.number_input(
                "Drift check every N",
                min_value=50,
                max_value=500,
                value=st.session_state.get("drift_check_every", 100),
                step=50,
            )
            st.session_state["drift_check_every"] = drift_check_every

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
        st.session_state["last_drift_check_n"] = 0
        st.session_state["last_drift_status"] = None
        if st.session_state.get("drift_enabled"):
            st.session_state["drift_detector"] = None
            st.session_state["alert_manager"] = None
        # Reset card history cache
        from collections import defaultdict
        st.session_state["card_history_cache"] = defaultdict(list)
        _create_scorer_and_tracker(
            default_threshold,
            num_f,
            cat_f,
            model_version=st.session_state.get("model_version", "v2"),
            target_encodings=st.session_state.get("target_encodings", {}),
            feature_meta=st.session_state.get("feature_meta", {}),
            window_size=window_size,
            history_every=history_every,
            min_samples=min_samples,
        )
        st.rerun()
    if run and not st.session_state.get("paused", False):
        _run_batch(batch_size)
        st.rerun()

    # Keep scorer/tracker aligned to the primary (review/cost) threshold.
    try:
        st.session_state["scorer"].threshold = float(threshold)
        st.session_state["metrics_tracker"].threshold = float(threshold)
    except Exception:
        pass

    _kpis(threshold)

    if st.session_state.get("dual_threshold", False):
        thr = st.session_state.get("thresholds", {}) or {}
        review_thr = float(thr.get("cost", threshold))
        block_thr = float(thr.get("f1", threshold))
        st.info(
            f"**Dual-threshold:** Approve (score < {review_thr:.2f}) → "
            f"**Review queue** ({review_thr:.2f} ≤ score < {block_thr:.2f}) → "
            f"Auto block (score ≥ {block_thr:.2f})"
        )

    r1, r2 = st.columns([1, 5])
    with r1:
        _drift_status_indicator()

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        [
            "Stream & high‑risk",
            "Metrics",
            "Confusion matrices",
            "Operational",
            "Drift & alerts",
            "Review queue",
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
        _tab_drift_alerts(threshold)
    with t6:
        _tab_review_queue(threshold)
    with t7:
        _tab_transaction_explorer(threshold)


if __name__ == "__main__":
    main()
