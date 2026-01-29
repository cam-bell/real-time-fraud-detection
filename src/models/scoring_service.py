"""Real-time scoring service wrapper."""

import time
import pickle
from datetime import datetime

import numpy as np

from src.features.feature_engineering import OnlineFeatureEngineer


class ScoringService:
    """Scores transactions in real-time using trained model."""

    def __init__(self, pipeline_path=None, threshold_path=None,
                 pipeline=None, threshold=None,
                 numerical_features=None, categorical_features=None,
                 feature_engineer=None,
                 simulate_latency_ms=None):
        """
        Args:
            pipeline_path: Path to pickled sklearn Pipeline.
            threshold_path: Path to pickled threshold.
            pipeline: Preloaded sklearn Pipeline (includes preprocessor).
            threshold: Decision threshold.
            numerical_features: Feature list used in training (if feature_engineer not provided).
            categorical_features: Feature list used in training (if feature_engineer not provided).
            feature_engineer: Pre-instantiated feature engineer (e.g., OnlineFeatureEngineerV3 for v3).
            simulate_latency_ms: Tuple (min_ms, max_ms) to add synthetic latency.
        """
        if pipeline_path:
            with open(pipeline_path, "rb") as f:
                self.pipeline = pickle.load(f)
        else:
            self.pipeline = pipeline

        if threshold_path:
            with open(threshold_path, "rb") as f:
                self.threshold = pickle.load(f)
        else:
            self.threshold = threshold

        if feature_engineer is not None:
            self.feature_engineer = feature_engineer
        else:
            if numerical_features is None or categorical_features is None:
                numerical_features = [] if numerical_features is None else numerical_features
                categorical_features = [] if categorical_features is None else categorical_features
            self.feature_engineer = OnlineFeatureEngineer(numerical_features, categorical_features)
        
        self.prediction_history = []
        self.latency_history = []
        self.simulate_latency_ms = simulate_latency_ms

    def predict_proba(self, transaction):
        """Score a single transaction and return fraud probability."""
        start_time = time.time()

        features_df = self.feature_engineer.engineer_features(transaction)
        fraud_probability = float(self.pipeline.predict_proba(features_df)[0, 1])

        # Update card history cache if using v3 engineer
        if hasattr(self.feature_engineer, 'update_history'):
            self.feature_engineer.update_history(transaction)

        elapsed_ms = (time.time() - start_time) * 1000
        if self.simulate_latency_ms:
            min_ms, max_ms = self.simulate_latency_ms
            elapsed_ms += float(np.random.uniform(min_ms, max_ms))

        self.latency_history.append(elapsed_ms)
        self.prediction_history.append({
            "timestamp": transaction.get("trans_date_trans_time", datetime.utcnow()),
            "score": fraud_probability,
            "amount": float(transaction.get("amt", 0) or 0),
            "latency_ms": float(elapsed_ms),
        })

        return fraud_probability

    def predict(self, transaction):
        """Return (probability, binary_prediction)."""
        p = self.predict_proba(transaction)
        threshold = 0.5 if self.threshold is None else float(self.threshold)
        pred = int(p >= threshold)
        return p, pred

    def get_stats(self):
        """Get scoring statistics."""
        if not self.latency_history:
            return {}
        return {
            "total_predictions": len(self.prediction_history),
            "avg_latency_ms": float(np.mean(self.latency_history)),
            "p95_latency_ms": float(np.percentile(self.latency_history, 95)),
            "p99_latency_ms": float(np.percentile(self.latency_history, 99)),
        }
