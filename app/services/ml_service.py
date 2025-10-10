from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .dl_service import FEATURE_COLUMNS, encode_features

MODEL_PATH = Path("xgb_model.pkl")

_model = None


def _ensure_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_from_raw(pclass: int, sex: str, fare: float, embarked: str) -> Dict[str, object]:
    feature_dict = encode_features(pclass, sex, fare, embarked)
    return predict_from_features(feature_dict)


def predict_from_features(feature_dict: Dict[str, float]) -> Dict[str, object]:
    model = _ensure_model()
    df = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]
    probabilities = (
        model.predict_proba(df)[0].tolist() if hasattr(model, "predict_proba") else []
    )

    return {
        "prediction": int(prediction),
        "probabilities": probabilities,
    }
