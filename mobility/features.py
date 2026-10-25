"""Calendar predictors and exact timestamp lookups for past rentals."""

import numpy as np
import pandas as pd

# Observed weather is omitted because it is not an archived weather forecast.
FEATURES = ["yr", "mnth", "hr", "holiday", "weekday", "workingday", "lag_24", "lag_168"]


def make_features(raw):
    """Look up 24h/168h earlier, leaving gaps missing instead of inventing counts."""
    df = raw.sort_values("timestamp").copy()
    if not df["timestamp"].is_unique:
        raise ValueError("Timestamps must be unique.")
    history = df.set_index("timestamp")["cnt"]
    for hours in [24, 168]:
        earlier = df["timestamp"] - pd.Timedelta(hours=hours)
        df[f"lag_{hours}"] = history.reindex(earlier).to_numpy()
    # A missing source hour is not evidence of zero rentals.
    return df[["timestamp"] + FEATURES + ["cnt"]].dropna().reset_index(drop=True)


def split_by_time(df):
    """Train through June 2012, validate July-September, test October-December."""
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    dates = pd.to_datetime(ordered["timestamp"])
    parts = (
        ordered[dates < "2012-07-01"].copy(),
        ordered[(dates >= "2012-07-01") & (dates < "2012-10-01")].copy(),
        ordered[dates >= "2012-10-01"].copy(),
    )
    if any(part.empty for part in parts):
        raise ValueError("Every chronological partition must contain rows.")
    if not parts[0]["timestamp"].max() < parts[1]["timestamp"].min():
        raise ValueError("Train and validation overlap.")
    if not parts[1]["timestamp"].max() < parts[2]["timestamp"].min():
        raise ValueError("Validation and test overlap.")
    return parts


def prediction_inputs(frame):
    """Validate the exact eight predictors used by training and scoring."""
    if (
        set(frame.columns) != set(FEATURES)
        or len(frame.columns) != len(FEATURES)
        or frame.empty
    ):
        raise ValueError(
            "Supply exactly the eight FEATURES; exclude targets and extra columns."
        )
    x = frame[FEATURES].copy()
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("All predictors must be finite and present.")
    ranges = {
        "yr": (0, 1),
        "mnth": (1, 12),
        "hr": (0, 23),
        "holiday": (0, 1),
        "weekday": (0, 6),
        "workingday": (0, 1),
        "lag_24": (0, np.inf),
        "lag_168": (0, np.inf),
    }
    for column, (low, high) in ranges.items():
        if not x[column].between(low, high).all():
            raise ValueError(f"Invalid predictor range: {column}")
        if not (x[column] % 1 == 0).all():
            raise ValueError(
                f"Expected integer calendar values or observed counts: {column}"
            )
    expected = (~x["weekday"].isin([0, 6]) & x["holiday"].eq(0)).astype(int)
    if not x["workingday"].eq(expected).all():
        raise ValueError("Inconsistent workingday/weekday/holiday.")
    return x
