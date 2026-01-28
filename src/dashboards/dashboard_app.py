"""Matplotlib dashboard for streaming monitoring."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class StreamingDashboard:
    """Real-time dashboard for monitoring the fraud detection system."""

    def __init__(self):
        self.fig = None
        self.axes = None

    def _ensure_fig(self, figsize):
        if self.fig is None:
            self.fig, self.axes = plt.subplots(3, 3, figsize=figsize)
            plt.ion()

    def create_dashboard(self, scorer, metrics_tracker, drift_detector, alert_manager,
                         transaction_history=None, figsize=(20, 12)):
        """Create or update the dashboard."""
        self._ensure_fig(figsize)
        axes = self.axes.flatten()

        for ax in axes:
            ax.clear()

        # Panel 1: Transaction stream
        if transaction_history:
            history_df = pd.DataFrame(transaction_history[-1000:])
            if not history_df.empty and {"timestamp", "score"}.issubset(history_df.columns):
                axes[0].plot(
                    pd.to_datetime(history_df["timestamp"]), 
                    history_df["score"],
                    alpha=0.6, linewidth=0.6, label="Score")
                fraud_mask = history_df.get("label", pd.Series([0] * len(history_df))) == 1
                if fraud_mask.any():
                    fraud_df = history_df[fraud_mask]
                    axes[0].scatter(pd.to_datetime(fraud_df["timestamp"]),
                                    fraud_df["score"], color="red", marker="x",
                                    s=50, label="True Fraud", zorder=5)
                axes[0].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Threshold")
                axes[0].set_title('Transaction Stream: Fraud Scores over Time\n(Simulated Transaction Time)')
                axes[0].set_xlabel("Time")
                axes[0].set_ylabel("Fraud Score")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis="x", rotation=45)

        # Panel 2: Performance metrics
        history_df = metrics_tracker.get_metric_history_df()
        if not history_df.empty and "timestamp" in history_df.columns:
            if "roc_auc" in history_df.columns:
                roc_df = history_df[history_df["roc_auc"].notna()].tail(20)
                if not roc_df.empty:
                    axes[1].plot(roc_df["timestamp"], roc_df["roc_auc"],
                             marker="o", markersize=3, label="ROC-AUC", linewidth=1.5)
            if "pr_auc" in history_df.columns:
                pr_df = history_df[history_df["pr_auc"].notna()]
                axes[1].plot(pr_df["timestamp"], pr_df["pr_auc"],
                             marker="s", markersize=3, label="PR-AUC", linewidth=1.5)
            axes[1].set_title('Performance Metrics over Time\n(System Computation Time)')
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Score")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[1].text(0.2, 0.5, "Insufficient data for ROC/PR-AUC",
                         fontsize=9, verticalalignment="center")
            axes[1].set_title("Performance Metrics over Time")
            axes[1].axis("off")

        # Panel 3: Precision/Recall/F1
        if not history_df.empty and "timestamp" in history_df.columns:
            for metric, marker in [("precision", "o"), ("recall", "s"), ("f1", "^")]:
                if metric in history_df.columns:
                    axes[2].plot(history_df["timestamp"], history_df[metric],
                                 marker=marker, markersize=3, label=metric.title(), linewidth=1.5)
            axes[2].set_title("Precision, Recall, F1 over Time")
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Score")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis="x", rotation=45)
        else:
            axes[2].text(0.2, 0.5, "Insufficient data for precision/recall",
                         fontsize=9, verticalalignment="center")
            axes[2].set_title("Precision, Recall, F1 over Time")
            axes[2].axis("off")

        # Panel 4: Drift detection events
        if drift_detector is not None and drift_detector.drift_history:
            drift_df = pd.DataFrame(drift_detector.drift_history)
            if "timestamp" in drift_df.columns:
                for drift_type in ["performance", "distribution", "feature"]:
                    mask = drift_df["drift_types"].apply(
                        lambda x: drift_type in x if isinstance(x, list) else False
                    )
                    if mask.any():
                        axes[3].scatter(pd.to_datetime(drift_df[mask]["timestamp"]),
                                        [1] * mask.sum(), label=f"{drift_type} drift",
                                        s=100, alpha=0.7)
                axes[3].set_title("Drift Detection Events")
                axes[3].set_xlabel("Time")
                axes[3].set_ylabel("Drift Detected")
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                axes[3].tick_params(axis="x", rotation=45)
        else:
            axes[3].text(0.2, 0.5, "No drift events yet/ Drift detector not enabled (MVP mode)",
                         fontsize=9, verticalalignment="center")
            axes[3].set_title("Drift Detection Events")
            axes[3].axis("off")

        # Panel 5: Alert log
        if alert_manager is not None:
            recent_alerts = alert_manager.get_recent_alerts(10)
            if recent_alerts:
                alert_text = []
                for alert in recent_alerts[:5]:
                    timestamp = alert.get("timestamp", datetime.utcnow())
                    alert_type = alert.get("type", "unknown")
                    severity = alert.get("severity", "info")
                    alert_text.append(f"{timestamp.strftime('%H:%M:%S')} [{severity.upper()}] {alert_type}")
                axes[4].text(0.05, 0.5, "\n".join(alert_text),
                            fontsize=9, verticalalignment="center", family="monospace")
            else:
                axes[4].text(0.2, 0.5, "No alerts",
                            fontsize=9, verticalalignment="center")
        else:
            axes[4].text(0.2, 0.5, "Alert manager not enabled (MVP mode)",
                            fontsize=9, verticalalignment="center")
        axes[4].set_title("Recent Alerts (Last 5)")
        axes[4].axis("off")

        # Panel 6: Confusion matrix
        latest_metrics = metrics_tracker.get_latest_metrics()
        if latest_metrics:
            scores, labels, predictions = metrics_tracker.get_current_window()
            if len(scores) > 0:
                cm = confusion_matrix(labels, predictions, labels=[0, 1])
                sns.heatmap(cm, annot=True, fmt="d", ax=axes[5], cmap="Blues", cbar=False)
                axes[5].set_title("Current Window Confusion Matrix")
                axes[5].set_xlabel("Predicted")
                axes[5].set_ylabel("Actual")
        else:
            axes[5].text(0.2, 0.5, "Insufficient data for confusion matrix",
                         fontsize=9, verticalalignment="center")
            axes[5].set_title("Current Window Confusion Matrix")
            axes[5].axis("off")

        # Panel 7: Current metrics summary
        if latest_metrics:
            metrics_text = (
                f"ROC-AUC: {latest_metrics.get('roc_auc', np.nan):.4f}\n"
                f"PR-AUC: {latest_metrics.get('pr_auc', np.nan):.4f}\n"
                f"Precision: {latest_metrics.get('precision', 0):.4f}\n"
                f"Recall: {latest_metrics.get('recall', 0):.4f}\n"
                f"F1-Score: {latest_metrics.get('f1', 0):.4f}\n"
                f"Accuracy: {latest_metrics.get('accuracy', 0):.4f}\n"
                f"Fraud Rate: {latest_metrics.get('fraud_rate', 0):.4f}\n"
                f"Current Window: {latest_metrics.get('window_size', 0)}\n"
                f"Total Processed: {len(metrics_tracker.labels)}"
            )
            axes[6].text(0.05, 0.5, metrics_text, fontsize=10,
                         verticalalignment="center", family="monospace")
            axes[6].set_title("Current Window Metrics")
            axes[6].axis("off")
        else:
            axes[6].text(0.2, 0.5, "Insufficient data",
                         fontsize=9, verticalalignment="center")
            axes[6].set_title("Current Window Metrics")
            axes[6].axis("off")

        # Panel 8: Fraud rate over time
        if not history_df.empty and "timestamp" in history_df.columns and "fraud_rate" in history_df.columns:
            axes[7].plot(history_df["timestamp"], history_df["fraud_rate"],
                         marker="o", markersize=3, color="red", linewidth=1.5)
            axes[7].set_title("Fraud Rate over Time")
            axes[7].set_xlabel("Time")
            axes[7].set_ylabel("Fraud Rate")
            axes[7].grid(True, alpha=0.3)
            axes[7].tick_params(axis="x", rotation=45)
        else:
            axes[7].text(0.2, 0.5, "Insufficient data for fraud rate",
                         fontsize=9, verticalalignment="center")
            axes[7].set_title('Fraud Rate over Time\n(Current System Time)')
            axes[7].axis("off")

        # Panel 9: Scoring service stats
        stats = scorer.get_stats()
        if stats:
            stats_text = (
                f"Total Predictions: {stats.get('total_predictions', 0)}\n"
                f"Avg Latency: {stats.get('avg_latency_ms', 0):.2f} ms\n"
                f"P95 Latency: {stats.get('p95_latency_ms', 0):.2f} ms\n"
                f"P99 Latency: {stats.get('p99_latency_ms', 0):.2f} ms"
            )
            axes[8].text(0.05, 0.5, stats_text, fontsize=10,
                         verticalalignment="center", family="monospace")
        else:
            axes[8].text(0.2, 0.5, "No scoring stats yet",
                         fontsize=9, verticalalignment="center")
        axes[8].set_title("Scoring Service Statistics")
        axes[8].axis("off")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def save_snapshot(self, filepath):
        """Save dashboard snapshot."""
        if self.fig is None:
            return
        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
