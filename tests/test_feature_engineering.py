from src.features.feature_engineering import OnlineFeatureEngineer


def test_online_feature_engineer_builds_expected_columns():
    engineer = OnlineFeatureEngineer(
        numerical_features=["amt", "hour", "distance", "amt_log"],
        categorical_features=["category", "state"],
    )

    row = {
        "amt": 250.0,
        "lat": 40.0,
        "long": -74.0,
        "merch_lat": 40.1,
        "merch_long": -74.1,
        "category": "shopping_net",
        "state": "NY",
        "trans_date_trans_time": "2024-01-01 23:15:00",
    }

    features = engineer.engineer_features(row)

    assert list(features.columns) == ["amt", "hour", "distance", "amt_log", "category", "state"]
    assert features.loc[0, "hour"] == 23
    assert features.loc[0, "amt"] == 250.0
    assert features.loc[0, "category"] == "shopping_net"
    assert features.loc[0, "distance"] > 0
    assert features.loc[0, "amt_log"] > 0
