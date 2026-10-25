"""Verify the public data and reject invalid hourly records."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
RAW_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


def validate_hourly(frame):
    """Return sorted records with a timestamp; never silently repair source data."""
    if list(frame.columns) != RAW_COLUMNS or frame.empty:
        raise ValueError("Expected the original nonempty UCI hourly schema.")
    df = frame.copy()
    numbers = df.drop(columns="dteday")
    if df.isna().any().any() or not np.isfinite(numbers.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite source values.")
    limits = {
        "instant": (1, np.inf),
        "season": (1, 4),
        "yr": (0, 1),
        "mnth": (1, 12),
        "hr": (0, 23),
        "holiday": (0, 1),
        "weekday": (0, 6),
        "workingday": (0, 1),
        "weathersit": (1, 4),
        "temp": (0, 1),
        "atemp": (0, 1),
        "hum": (0, 1),
        "windspeed": (0, 1),
        "casual": (0, np.inf),
        "registered": (0, np.inf),
        "cnt": (0, np.inf),
    }
    for column, (low, high) in limits.items():
        if not df[column].between(low, high).all():
            raise ValueError(f"Out-of-range source values: {column}")
        if column not in ["temp", "atemp", "hum", "windspeed"]:
            if not (df[column] % 1 == 0).all():
                raise ValueError(f"Expected integer values: {column}")
    dates = pd.to_datetime(df["dteday"], format="%Y-%m-%d", errors="raise")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("dteday must contain dates without times.")
    df["timestamp"] = dates + pd.to_timedelta(df["hr"], unit="h")
    if not df["instant"].is_unique or not df["timestamp"].is_unique:
        raise ValueError("Duplicate record identifiers or date/hour pairs.")
    if not (df["casual"] + df["registered"]).eq(df["cnt"]).all():
        raise ValueError("casual + registered must equal cnt.")
    if not (dates.dt.month.eq(df["mnth"]) & (dates.dt.year - 2011).eq(df["yr"])).all():
        raise ValueError("Calendar month/year do not match the date.")
    if not ((dates.dt.dayofweek + 1) % 7).eq(df["weekday"]).all():
        raise ValueError("Weekday does not match the date (Sunday = 0).")
    expected_workday = (~df["weekday"].isin([0, 6]) & df["holiday"].eq(0)).astype(int)
    if not expected_workday.eq(df["workingday"]).all():
        raise ValueError("Working-day flag conflicts with weekday/holiday.")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_data():
    """Check the fixed source fingerprint before reading any records."""
    path = ROOT / "data" / "hour.csv"
    source = json.loads((ROOT / "data" / "source.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(
            "Source checksum mismatch. Restore the original public hour.csv."
        )
    df = validate_hourly(pd.read_csv(path))
    if len(df) != source["rows"]:
        raise ValueError("Unexpected source row count.")
    return df


def invalidate_outputs(stage):
    """Remove only known downstream results before a stage is rerun."""
    groups = {
        1: [
            "data_report.json",
            "hourly_summary.csv",
            "hourly_demand.png",
            "hourly.db",
            "train.csv",
            "validation.csv",
            "test.csv",
        ],
        2: [
            "model.joblib",
            "metrics.json",
            "validation_predictions.csv",
            "test_predictions.csv",
        ],
        3: ["capacity_validation.csv", "capacity_test.csv", "capacity.json"],
        4: [
            "monthly_errors.csv",
            "error_review.png",
            "example_input.csv",
            "example_prediction.csv",
        ],
    }
    OUTPUT.mkdir(exist_ok=True)
    for number, names in groups.items():
        if number >= stage:
            for name in names:
                (OUTPUT / name).unlink(missing_ok=True)
    executed = OUTPUT / "executed"
    if executed.exists():
        for path in executed.glob("*.ipynb"):
            if int(path.name[:2]) >= stage:
                path.unlink()
