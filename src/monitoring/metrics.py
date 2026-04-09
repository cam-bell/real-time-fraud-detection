"""Streaming metrics tracking."""

from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsTracker:
    """Tracks performance metrics over sliding windows."""

    def __init__(self, window_size=1000, time_window_minutes=None,
                 threshold=0.5, history_every=100, min_samples=100):
        self.window_size = int(window_size)
        self.time_window = timedelta(minutes=time_window_minutes) if time_window_minutes else None
        self.threshold = float(threshold)
        self.history_every = int(history_every)
        self.min_samples = int(min_samples)

        self.predictions = deque(maxlen=self.window_size * 2)
        self.labels = deque(maxlen=self.window_size * 2)
        self.scores = deque(maxlen=self.window_size * 2)
        self.timestamps = deque(maxlen=self.window_size * 2)
        self.metric_history = []
        self._update_count = 0

    def update(self, transaction, score, label):
        """Update with new prediction."""
        transaction_timestamp = pd.to_datetime(
            transaction.get("trans_date_trans_time", datetime.now(timezone.utc))
        )
        prediction = 1 if score >= self.threshold else 0

        self.predictions.append(prediction)
        self.labels.append(int(label))
        self.scores.append(float(score))
        self.timestamps.append(transaction_timestamp)
        self._update_count += 1

        if self._update_count % self.history_every == 0:
            metrics = self.compute_metrics()
            if metrics:
                # Use CURRENT TIME for when metric was computed (system time)
                metrics["timestamp"] = datetime.now(timezone.utc) # System time
                # Optionally keep transaction time for reference
                metrics["transaction_timestamp"] = transaction_timestamp  # Historical data time
                self.metric_history.append(metrics)

    def get_current_window(self):
        """Get current window of data based on window type."""
        if self.time_window:
            if not self.timestamps:
                return [], [], []
            cutoff_time = self.timestamps[-1] - self.time_window
            indices = [i for i, ts in enumerate(self.timestamps) if ts >= cutoff_time]
            if not indices:
                return [], [], []
            return (
                [self.scores[i] for i in indices],
                [self.labels[i] for i in indices],
                [self.predictions[i] for i in indices],
            )

        n = min(self.window_size, len(self.scores))
        return (
            list(self.scores)[-n:],
            list(self.labels)[-n:],
            list(self.predictions)[-n:],
        )

    def compute_metrics(self):
        """Compute metrics for current window."""
        scores, labels, predictions = self.get_current_window()
        if len(scores) < self.min_samples:
            return None

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        preds_arr = np.array(predictions)

        metrics = {
            "precision": precision_score(labels_arr, preds_arr, zero_division=0),
            "recall": recall_score(labels_arr, preds_arr, zero_division=0),
            "f1": f1_score(labels_arr, preds_arr, zero_division=0),
            "accuracy": accuracy_score(labels_arr, preds_arr),
            "window_size": len(scores_arr),
            "fraud_rate": float(labels_arr.mean()),
            "predicted_fraud_rate": float(preds_arr.mean()),
        }

        if np.unique(labels_arr).size >= 2:
            metrics["roc_auc"] = roc_auc_score(labels_arr, scores_arr)
            metrics["pr_auc"] = average_precision_score(labels_arr, scores_arr)
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan

        return metrics

    def get_latest_metrics(self):
        """Get most recent metrics."""
        return self.compute_metrics()

    def get_metric_history_df(self):
        """Get metric history as DataFrame."""
        if not self.metric_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.metric_history)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        return df
