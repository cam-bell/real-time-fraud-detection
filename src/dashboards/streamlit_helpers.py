"""Helpers for Streamlit fraud dashboard: data prep, load, segment."""

from pathlib import Path
import json
import pickle
import os

import numpy as np
import pandas as pd

from src.utils.helpers import calculate_distance
from src.features.feature_engineering import add_eda_features


def get_feature_lists(include_eda_features=True):
    """Feature lists for modeling. Matches notebook."""
    base_numerical = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "distance",
        "amt_log",
    ]
    eda_numerical = [
        "amt_high_risk",
        "amt_very_high_risk",
        "amt_medium_risk",
        "is_late_night",
        "is_high_fraud_hour",
        "is_low_fraud_hour",
        "is_high_risk_category",
        "is_medium_risk_category",
        "high_amt_late_night",
        "high_risk_cat_high_amt",
        "weekend_high_amt",
    ]
    numerical = base_numerical + (eda_numerical if include_eda_features else [])
    categorical = ["category", "gender", "state", "merchant"]
    return numerical, categorical


def engineer_basic_features(df):
    """Temporal and distance features. Mirrors notebook."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["day_of_month"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["distance"] = df.apply(
        lambda r: calculate_distance(
            r["lat"], r["long"], r["merch_lat"], r["merch_long"]
        ),
        axis=1,
    )
    df["amt_log"] = np.log1p(df["amt"])
    return df


def find_segment_with_fraud(df, min_fraud=5, segment_size=2000):
    """First segment of segment_size with >= min_fraud fraud cases; else full df."""
    if "is_fraud" not in df.columns:
        return df.iloc[:segment_size].reset_index(drop=True)
    for start in range(0, len(df), segment_size):
        seg = df.iloc[start : start + segment_size]
        if seg["is_fraud"].sum() >= min_fraud:
            return seg.reset_index(drop=True)
    return df.iloc[: min(segment_size, len(df))].reset_index(drop=True)


def _data_dir():
    base = os.environ.get("FRAUD_DATA_DIR")
    if base:
        return Path(base)
    return Path("data")


def _models_dir():
    base = os.environ.get("FRAUD_MODELS_DIR")
    if base:
        return Path(base)
    return Path("models")


def _load_threshold(models_dir):
    for name in ("threshold.pkl", "optimal_threshold.pkl"):
        p = models_dir / name
        if p.exists():
            with open(p, "rb") as f:
                return float(pickle.load(f))
    meta = models_dir / "model_metadata.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    return 0.5


def _load_metadata(models_dir, filename="model_metadata.json"):
    p = models_dir / filename
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_threshold_pair(models_dir, *, meta, version):
    """
    Return (cost_threshold, f1_threshold) for dual-threshold workflow.

    - cost (lower): scores >= this go to manual review queue (or alerts).
    - f1 (higher): scores >= this are auto-blocked; between cost and f1 = review queue only.
    - v2: uses existing single threshold as both (backwards-compatible).
    - v3: tries dedicated files first, then metadata keys.
    """
    if version != "v3":
        t = _load_threshold(models_dir)
        return float(t), float(t)

    # Prefer explicit artifacts if present
    p_cost = models_dir / "threshold_cost_v3.pkl"
    p_f1 = models_dir / "threshold_f1_v3.pkl"
    if p_cost.exists() and p_f1.exists():
        with open(p_cost, "rb") as f:
            cost_t = float(pickle.load(f))
        with open(p_f1, "rb") as f:
            f1_t = float(pickle.load(f))
        return cost_t, f1_t

    # Fall back to metadata
    cost_t = meta.get("threshold_cost")
    f1_t = meta.get("threshold_f1")
    if cost_t is not None and f1_t is not None:
        return float(cost_t), float(f1_t)

    # Last resort: use legacy threshold semantics
    t = _load_threshold(models_dir)
    return float(t), float(t)


def available_model_versions(models_dir=None):
    """Return available model versions based on artifact presence."""
    models_dir = models_dir or _models_dir()
    versions = ["v2"]
    if (models_dir / "fraud_detection_pipeline_v3.pkl").exists():
        versions.append("v3")
    return versions


def load_model_threshold_data(
    *,
    use_segment=True,
    min_fraud=5,
    segment_size=2000,
    model_version="v2",
):
    """
    Load pipeline, threshold, test data; build prepped segment.
    Returns (pipeline, thresholds, segment, num_features, cat_features, target_encodings, feature_meta).

    thresholds: dict with "cost" (review-threshold) and "f1" (block-threshold).
    Dashboard uses cost for review queue (manual review), f1 for auto-block.
    For v3, also returns target_encodings dict and feature metadata.
    """
    models_dir = _models_dir()
    data_dir = _data_dir()

    pipeline_name = (
        "fraud_detection_pipeline_v3.pkl" if model_version == "v3" else "fraud_detection_pipeline.pkl"
    )
    with open(models_dir / pipeline_name, "rb") as f:
        pipeline = pickle.load(f)
    meta_name = "model_metadata_v3.json" if model_version == "v3" else "model_metadata.json"
    meta = _load_metadata(models_dir, filename=meta_name)
    cost_t, f1_t = _load_threshold_pair(models_dir, meta=meta, version=model_version)
    thresholds = {"cost": float(cost_t), "f1": float(f1_t)}
    cat_lists = meta.get("category_risk_lists", {})
    high = cat_lists.get("high_risk", ["shopping_net", "grocery_pos", "misc_net", "home"])
    medium = cat_lists.get("medium_risk", ["food_dining", "kids_pets", "health_fitness"])

    df = pd.read_csv(data_dir / "fraudTest.csv")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    if "merchant" not in df.columns and "merch_name" in df.columns:
        df["merchant"] = df["merch_name"]

    df = engineer_basic_features(df)
    df = add_eda_features(
        df,
        high_risk_categories=high,
        medium_risk_categories=medium,
    )
    seg = find_segment_with_fraud(df, min_fraud=min_fraud, segment_size=segment_size) if use_segment else df.iloc[:segment_size].reset_index(drop=True)
    
    # Load feature lists and v3-specific artifacts
    target_encodings = {}
    feature_meta = {}
    
    if model_version == "v3":
        # Load target encodings
        encodings_path = models_dir / "target_encodings_v3.json"
        if encodings_path.exists():
            with open(encodings_path, encoding="utf-8") as f:
                target_encodings = json.load(f)
                # Extract global_mean if stored
                if "_global_mean" not in target_encodings:
                    encoding_info = meta.get("target_encoding", {})
                    target_encodings["_global_mean"] = float(encoding_info.get("global_mean", 0.0))
        
        # Get v3 feature lists from metadata
        feature_lists = meta.get("feature_lists", {})
        num = feature_lists.get("numerical", [])
        cat = feature_lists.get("categorical_target_encoded", [])
        # Ensure gender_encoded is in numerical features (as it was during training)
        if "gender_encoded" not in num:
            num = num + ["gender_encoded"]
        feature_meta = {
            "all_model_features": feature_lists.get("all_model_features", num + cat),
        }
    else:
        num, cat = get_feature_lists(include_eda_features=True)
    
    return pipeline, thresholds, seg, num, cat, target_encodings, feature_meta
