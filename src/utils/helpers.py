"""Utility helpers for the fraud real-time pipeline."""

import math


def calculate_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers between two lat/lon points."""
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return 0.0

    # Guard against missing coordinates
    if any(math.isnan(x) for x in [lat1, lon1, lat2, lon2]):
        return 0.0

    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c
