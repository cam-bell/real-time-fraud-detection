"""Streaming transaction simulator."""

import time

import pandas as pd


class StreamSimulator:
    """Iterate over transactions with optional time scaling."""

    def __init__(self, df, speed="fast", time_scale=60.0, max_sleep=0.5):
        """
        Args:
            df: DataFrame with transaction data.
            speed: "fast", "real-time", "custom" or a numeric time_scale.
            time_scale: Real seconds per data minute when speed == "real-time".
            max_sleep: Cap sleep per event for practical playback.
        """
        self.df = df.copy()
        if "trans_date_trans_time" in self.df.columns:
            self.df["trans_date_trans_time"] = pd.to_datetime(self.df["trans_date_trans_time"])
            self.df = self.df.sort_values("trans_date_trans_time")

        if isinstance(speed, (int, float)):
            self.speed = "real-time"
            self.time_scale = float(speed)
        else:
            self.speed = speed
            self.time_scale = float(time_scale)

        self.max_sleep = float(max_sleep)
        self._records = self.df.to_dict("records")

    def __iter__(self):
        prev_ts = None
        for txn in self._records:
            curr_ts = txn.get("trans_date_trans_time")
            if self.speed in {"real-time", "custom"} and prev_ts is not None:
                try:
                    delta_seconds = (curr_ts - prev_ts).total_seconds()
                except Exception:
                    delta_seconds = 0.0

                if self.speed == "real-time":
                    sleep_seconds = max(0.0, min(self.max_sleep, delta_seconds / max(self.time_scale, 1e-6)))
                else:
                    sleep_seconds = self.max_sleep

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            prev_ts = curr_ts
            yield txn
