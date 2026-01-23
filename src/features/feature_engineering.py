"""Feature engineering for online scoring."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.helpers import calculate_distance


class OnlineFeatureEngineer:
    """Applies feature engineering to single transactions for online scoring."""

    def __init__(self, numerical_features, categorical_features,
                 high_risk_categories=None, medium_risk_categories=None):
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)

        if high_risk_categories is None:
            self.high_risk_categories = ["shopping_net", "grocery_pos", "misc_net", "home"]
        else:
            self.high_risk_categories = list(high_risk_categories)

        if medium_risk_categories is None:
            self.medium_risk_categories = ["food_dining", "kids_pets", "health_fitness"]
        else:
            self.medium_risk_categories = list(medium_risk_categories)

    def engineer_features(self, transaction):
        """Engineer features from a single transaction."""
        if isinstance(transaction, pd.Series):
            txn = transaction.to_dict()
        else:
            txn = dict(transaction)

        # Parse timestamp if string
        if isinstance(txn.get("trans_date_trans_time"), str):
            txn["trans_date_trans_time"] = pd.to_datetime(txn["trans_date_trans_time"])

        dt = pd.to_datetime(txn.get("trans_date_trans_time", datetime.utcnow()))
        txn["hour"] = dt.hour
        txn["day_of_week"] = dt.dayofweek
        txn["day_of_month"] = dt.day
        txn["month"] = dt.month
        txn["is_weekend"] = 1 if dt.dayofweek >= 5 else 0

        # Distance feature
        txn["distance"] = calculate_distance(
            txn.get("lat", 0),
            txn.get("long", 0),
            txn.get("merch_lat", 0),
            txn.get("merch_long", 0),
        )

        # Amount transforms
        amt = float(txn.get("amt", 0) or 0)
        txn["amt_log"] = np.log1p(amt)

        # Amount risk bins
        txn["amt_high_risk"] = 1 if amt > 500 else 0
        txn["amt_very_high_risk"] = 1 if amt > 1000 else 0
        txn["amt_medium_risk"] = 1 if 100 < amt <= 500 else 0

        # Time indicators
        hour = txn.get("hour", 0)
        txn["is_late_night"] = 1 if hour in [22, 23, 0, 1, 2, 3] else 0
        txn["is_high_fraud_hour"] = 1 if hour in [22, 23, 0, 1, 2, 3] else 0
        txn["is_low_fraud_hour"] = 1 if 6 <= hour <= 14 else 0

        # Category risk
        category = txn.get("category", "")
        txn["is_high_risk_category"] = 1 if category in self.high_risk_categories else 0
        txn["is_medium_risk_category"] = 1 if category in self.medium_risk_categories else 0

        # Interaction features
        txn["high_amt_late_night"] = 1 if (amt > 500 and txn["is_late_night"] == 1) else 0
        txn["high_risk_cat_high_amt"] = 1 if (txn["is_high_risk_category"] == 1 and amt > 500) else 0
        txn["weekend_high_amt"] = 1 if (txn["is_weekend"] == 1 and amt > 500) else 0

        # Build feature vector in training order
        feature_vector = []
        for feat in self.numerical_features + self.categorical_features:
            feature_vector.append(txn.get(feat, 0))

        return pd.DataFrame([feature_vector], columns=self.numerical_features + self.categorical_features)

    def prepare_for_model(self, transaction, preprocessor):
        """Engineer features and apply preprocessing."""
        features_df = self.engineer_features(transaction)
        return preprocessor.transform(features_df)

def add_eda_features(df, 
                     high_risk_categories=None, 
                     medium_risk_categories=None,
                     version='v1'):
    """
    Add EDA-based features to dataframe (batch processing).
    
    This is the batch version for training data preparation.
    For single-transaction feature engineering, use OnlineFeatureEngineer.
    """
    # Default category lists
    if high_risk_categories is None:
        high_risk_categories = ['shopping_net', 'misc_net', 'grocery_pos']
    if medium_risk_categories is None:
        medium_risk_categories = ['shopping_pos', 'gas_transport']

    # Ensure hour is available
    if 'hour' not in df.columns:
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date_trans_time'].dt.hour
        else:
            raise ValueError("Need 'hour' or 'trans_date_trans_time' column")
    
    # Amount risk bins
    df['amt_high_risk'] = (df['amt'] > 500).astype(int)
    df['amt_very_high_risk'] = (df['amt'] > 1000).astype(int)
    df['amt_medium_risk'] = ((df['amt'] > 100) & (df['amt'] <= 500)).astype(int)
    
    # Time indicators
    df['is_late_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df['is_high_fraud_hour'] = df['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df['is_low_fraud_hour'] = df['hour'].between(6, 14, inclusive='both').astype(int)
    
    # Category risk levels
    df['is_high_risk_category'] = df['category'].isin(high_risk_categories).astype(int)
    df['is_medium_risk_category'] = df['category'].isin(medium_risk_categories).astype(int)
    
    # Interaction features
    df['high_amt_late_night'] = ((df['amt'] > 500) & (df['is_late_night'] == 1)).astype(int)
    df['high_risk_cat_high_amt'] = ((df['is_high_risk_category'] == 1) & (df['amt'] > 500)).astype(int)
    
    # Ensure is_weekend exists
    if 'is_weekend' not in df.columns:
        if 'day_of_week' in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        elif 'trans_date_trans_time' in df.columns:
            df['is_weekend'] = (pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek >= 5).astype(int)
    
    df['weekend_high_amt'] = ((df['is_weekend'] == 1) & (df['amt'] > 500)).astype(int)
    
    return df