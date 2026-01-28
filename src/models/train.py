import pandas as pd
from datetime import datetime
import time

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import add_eda_features

categorical_features = [
    'category', 'subcategory', 'mcc', 'first', 'last', 'zip', 'state', 'city', 'job',
    'gender', 'dob_year', 'dob_month', 'dob_day', 'trans_num', 'trans_date_trans_time',
    'trans_date', 'trans_time', 'unix_time', 'merch_name', 'is_fraud'
]
numerical_features = [
    'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
    'distance', 'amt_log',
    'amt_high_risk', 'amt_very_high_risk', 'amt_medium_risk',
    'is_late_night', 'is_high_fraud_hour', 'is_low_fraud_hour',
    'is_high_risk_category', 'is_medium_risk_category',
    'high_amt_late_night', 'high_risk_cat_high_amt', 'weekend_high_amt'
]
class AutoRetrainer:
    """Handles automatic model retraining when drift is detected"""
    
    def __init__(self, original_train_df, numerical_features, categorical_features):
        self.original_train_df = original_train_df
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.retraining_history = []
        self.recent_transactions = []
        self.max_recent_transactions = 10000  # Keep last 10k for retraining
        
    def add_transaction(self, transaction, label):
        """Add transaction to recent transactions pool"""
        self.recent_transactions.append({
            'transaction': transaction,
            'label': label,
            'timestamp': pd.to_datetime(transaction.get('trans_date_trans_time', datetime.now()))
        })
        # Keep only recent transactions
        if len(self.recent_transactions) > self.max_recent_transactions:
            self.recent_transactions = self.recent_transactions[-self.max_recent_transactions:]
    
    def retrain(self, n_recent=5000):
        """Retrain model on combined historical + recent data"""
        print(f"\n🔄 Starting model retraining with {n_recent} recent transactions...")
        start_time = time.time()
        
        # Get recent transactions
        recent = self.recent_transactions[-n_recent:] if len(self.recent_transactions) >= n_recent else self.recent_transactions
        
        if len(recent) < 100:
            print("⚠ Not enough recent transactions for retraining")
            return None
        
        # Combine with original training data
        
        # Fixed - ensure all transactions are dicts:
        transactions_list = []
        for r in recent:
            txn = r['transaction']
            # Convert Series to dict if needed
            if isinstance(txn, pd.Series):
                txn = txn.to_dict()
            elif not isinstance(txn, dict):
                txn = dict(txn) # Convert to dict if it's some other type
            transactions_list.append(txn)

        recent_df = pd.DataFrame(transactions_list)
        recent_df['is_fraud'] = [r['label'] for r in recent]
        
        # Apply feature engineering
        recent_df = add_eda_features(recent_df, 
                           high_risk_categories=['shopping_net', 'grocery_pos', 'misc_net', 'home'],
                           medium_risk_categories=['food_dining', 'kids_pets', 'health_fitness'],
                           version='v2')
        
        # Combine datasets
        combined_train = pd.concat([
            self.original_train_df,
            recent_df
        ], ignore_index=True)
        
        # Prepare features
        X_combined = combined_train[numerical_features + categorical_features].copy()
        y_combined = combined_train['is_fraud'].copy()
        
        # Create and train new model
        new_preprocessor = ColumnTransformer([
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')
        
        new_model = Pipeline([
            ('preprocessor', new_preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=len(y_combined[y_combined==0]) / len(y_combined[y_combined==1]),
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            ))
        ])
        
        new_model.fit(X_combined, y_combined)
        
        retraining_time = time.time() - start_time
        
        retrain_record = {
            'timestamp': datetime.now(),
            'n_recent_transactions': len(recent),
            'n_total_training': len(combined_train),
            'retraining_time_seconds': retraining_time
        }
        
        self.retraining_history.append(retrain_record)
        print(f"✓ Retraining complete in {retraining_time:.2f} seconds")
        
        return new_model, new_preprocessor
    
    