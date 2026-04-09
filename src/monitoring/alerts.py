"""Alert management for streaming monitoring."""

from typing import Any


from datetime import datetime, timezone


class AlertManager:
    """Collects and surfaces operational alerts."""

    def __init__(self, high_risk_threshold=0.9, performance_drop_threshold=0.05):
        self.high_risk_threshold = float(high_risk_threshold)
        self.performance_drop_threshold = float(performance_drop_threshold)
        self.alerts = []
        self.baseline_roc_auc = None

    def set_baseline_performance(self, roc_auc):
        self.baseline_roc_auc = None if roc_auc is None else float(roc_auc)

    def add_alert(self, alert_type, severity, message, metadata=None):
        alert = {
            "timestamp": datetime.now(timezone.utc),
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
        }
        self.alerts.append(alert)
        return alert

    def get_recent_alerts(self, n=10):
        return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:n]

    def check_high_risk_transaction(self, transaction, score):
        if score >= self.high_risk_threshold:
            return self.add_alert(
                "high_risk_transaction",
                "high",
                f"High risk transaction score={score:.3f}",
                {
                    "transaction_id": transaction.get("trans_num"),
                    "amount": transaction.get("amt"),
                    "merchant": transaction.get("merchant"),
                },
            )
        return None

    def check_drift_alert(self, drift_status):
        if drift_status.get("drift_detected"):
            return self.add_alert(
                "drift_detected",
                "medium",
                f"Drift detected: {drift_status.get('drift_types', [])}",
                drift_status.get("details", {}),
            )
        return None

    def check_performance_degradation(self, current_roc_auc):
        if self.baseline_roc_auc is None or current_roc_auc is None:
            return None
        drop = self.baseline_roc_auc - current_roc_auc
        if drop >= self.performance_drop_threshold:
            return self.add_alert(
                "performance_degradation",
                "medium",
                f"ROC-AUC drop {drop:.3f}",
                {"baseline_roc_auc": self.baseline_roc_auc, "current_roc_auc": current_roc_auc},
            )
        return None

    def check_model_retrained(self, retrain_info):
        return self.add_alert(
            "model_retrained",
            "info",
            "Model retrained",
            retrain_info or {},
        )
