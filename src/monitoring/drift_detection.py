"""Simple drift detection utilities."""

from collections import deque
from datetime import datetime, timezone

import numpy as np
from scipy.stats import ks_2samp


class DriftDetector:
    """Track distribution drift and performance degradation."""

    def __init__(self, reference_window=500, detection_window=500,
                 ks_alpha=0.01, ks_stat_threshold=0.1,
                 performance_drop_threshold=0.05,
                 cooldown_after_retrain=500):
        
        self.reference_window = int(reference_window)
        self.detection_window = int(detection_window)
        self.ks_alpha = float(ks_alpha)
        self.ks_stat_threshold = float(ks_stat_threshold)
        self.performance_drop_threshold = float(performance_drop_threshold)
        
        self.cooldown_after_retrain = int(cooldown_after_retrain)
        self.last_retrain_transaction = None
        self._update_count = 0


        self.reference_scores = deque(maxlen=self.reference_window)
        self.reference_features = {}
        self.detection_scores = deque(maxlen=self.detection_window)
        self.detection_features = {}
        self.drift_history = []
        self.baseline_roc_auc = None
        

    def _append_features(self, store, features_dict):
        for key, value in features_dict.items():
            if key not in store:
                store[key] = deque(maxlen=self.reference_window if store is self.reference_features else self.detection_window)
            store[key].append(float(value))

    def update_reference(self, score, features_dict):
        self.reference_scores.append(float(score))
        self._append_features(self.reference_features, features_dict)

    def update_detection(self, score, features_dict):
        self.detection_scores.append(float(score))
        self._append_features(self.detection_features, features_dict)

    def set_baseline_performance(self, roc_auc):
        self.baseline_roc_auc = None if roc_auc is None else float(roc_auc)

    def update_reference_from_detection(self):
        self.reference_scores = deque(self.detection_scores, maxlen=self.reference_window)
        self.reference_features = {
            key: deque(values, maxlen=self.reference_window)
            for key, values in self.detection_features.items()
        }

    def _ks_test(self, ref_vals, det_vals):
        if len(ref_vals) < 30 or len(det_vals) < 30:
            return None
        statistic, p_value = ks_2samp(ref_vals, det_vals)
        return statistic, p_value

    def check_drift(self, current_roc_auc=None, transaction_count=None):
        """Check for drift and return drift status"""
        drift_types = []
        details = {}

        # Skip if in cooldown
        if (self.last_retrain_transaction is not None and
            transaction_count is not None and
            transaction_count - self.last_retrain_transaction < self.cooldown_after_retrain):
            return {"drift_detected": False, "drift_types": [], "details": {}}

        # Performance drift
        if current_roc_auc is not None and self.baseline_roc_auc is not None:
            if self.baseline_roc_auc - current_roc_auc >= self.performance_drop_threshold:
                drift_types.append("performance")
                details["performance_drop"] = self.baseline_roc_auc - current_roc_auc

        # Score distribution drift
        ks_result = self._ks_test(list(self.reference_scores), list(self.detection_scores))
        if ks_result:
            statistic, p_value = ks_result
            if p_value < self.ks_alpha and statistic >= self.ks_stat_threshold:
                drift_types.append("distribution")
                details["score_ks_stat"] = statistic
                details["score_ks_p"] = p_value

        # Feature drift
        feature_drifts = []
        for feature, det_values in self.detection_features.items():
            ref_values = self.reference_features.get(feature, [])
            ks_result = self._ks_test(list(ref_values), list(det_values))
            if ks_result:
                statistic, p_value = ks_result
                if p_value < self.ks_alpha and statistic >= self.ks_stat_threshold:
                    feature_drifts.append({"feature": feature, "ks_stat": statistic, "p_value": p_value})

        if feature_drifts:
            drift_types.append("feature")
            details["feature_drifts"] = feature_drifts

        drift_status = {
            "timestamp": datetime.now(timezone.utc),
            "drift_detected": bool(drift_types),
            "drift_types": drift_types,
            "details": details,
        }

        if drift_status["drift_detected"]:
            self.drift_history.append(drift_status)

        return drift_status
    
    def rebuild_reference_from_new_model(self, new_transactions_scored):
        """Clear and rebuild reference window with new model scores
        Args:
            new_transactions_scored: List of (score, features_dict) tuples
        """
        self.reference_scores.clear()
        self.reference_features.clear()
        for score, features_dict in new_transactions_scored:
            self.update_reference(score, features_dict)
    
    def mark_retrain(self, transaction_count):
        """Mark that retraining occurred at this transaction count."""
        self.last_retrain_transaction = transaction_count
        # Also clear detection window to start fresh
        self.detection_scores.clear()
        self.detection_features.clear()
