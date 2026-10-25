"""Reload the local model and apply the same predictor rules as training."""

import joblib
import numpy as np

from mobility.data import OUTPUT
from mobility.features import FEATURES, prediction_inputs


def predict_saved(frame, model_path=None):
    """Load only trusted model files; joblib deserialization can execute code."""
    x = prediction_inputs(frame)
    path = OUTPUT / "model.joblib" if model_path is None else model_path
    artifact = joblib.load(path)
    if artifact["features"] != FEATURES:
        raise ValueError("Model feature list differs. Rerun notebook 02.")
    if artifact["selected_model"] == "last_week":
        return x["lag_168"].to_numpy(dtype=float)
    if artifact["selected_model"] != "gradient_boosting":
        raise ValueError("Unknown saved model choice.")
    return np.maximum(0, artifact["estimator"].predict(x))
