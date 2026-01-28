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


def _load_metadata(models_dir):
    p = models_dir / "model_metadata.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_model_threshold_data(
    *,
    use_segment=True,
    min_fraud=5,
    segment_size=2000,
):
    """
    Load pipeline, threshold, test data; build prepped segment.
    Returns (pipeline, threshold, segment, num_features, cat_features).
    """
    models_dir = _models_dir()
    data_dir = _data_dir()

    with open(models_dir / "fraud_detection_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    threshold = _load_threshold(models_dir)
    meta = _load_metadata(models_dir)
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
    num, cat = get_feature_lists(include_eda_features=True)
    return pipeline, threshold, seg, num, cat
