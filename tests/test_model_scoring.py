import numpy as np
import pandas as pd

from src.models.scoring_service import ScoringService


class DummyPipeline:
    def predict_proba(self, features_df):
        assert isinstance(features_df, pd.DataFrame)
        return np.array([[0.2, 0.8]])


def test_scoring_service_predict_respects_threshold():
    service = ScoringService(
        pipeline=DummyPipeline(),
        threshold=0.5,
        numerical_features=["amt", "hour", "distance", "amt_log"],
        categorical_features=["category", "state"],
    )

    probability, prediction = service.predict(
        {
            "amt": 800.0,
            "lat": 40.0,
            "long": -74.0,
            "merch_lat": 40.1,
            "merch_long": -74.1,
            "category": "shopping_net",
            "state": "NY",
            "trans_date_trans_time": "2024-01-01 23:15:00",
        }
    )

    assert probability == 0.8
    assert prediction == 1
