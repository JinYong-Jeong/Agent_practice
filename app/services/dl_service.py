from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from train.dltrain import BasicMLP

MODEL_PATH = Path("dl_model.pth")
SCALER_PATH = Path("scaler.pkl")
FEATURE_COLUMNS = [
    "Fare",
    "Pclass_2",
    "Pclass_3",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]
CLASS_LABELS = {0: 'deceased', 1: 'survived'}

_model: BasicMLP | None = None
_scaler: StandardScaler | None = None


def _ensure_artifacts() -> Tuple[BasicMLP, StandardScaler]:
    global _model, _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _model is None:
        model = BasicMLP(len(FEATURE_COLUMNS))
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
    return _model, _scaler


def encode_features(pclass: int, sex: str, fare: float, embarked: str) -> Dict[str, float]:
    return {
        "Fare": fare,
        "Pclass_2": 1.0 if pclass == 2 else 0.0,
        "Pclass_3": 1.0 if pclass == 3 else 0.0,
        "Sex_male": 1.0 if sex == "male" else 0.0,
        "Embarked_Q": 1.0 if embarked == "Q" else 0.0,
        "Embarked_S": 1.0 if embarked == "S" else 0.0,
    }


def predict_from_dict(feature_dict: Dict[str, float]) -> Dict[str, object]:
    model, scaler = _ensure_artifacts()
    row = np.array([[feature_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    row_scaled = scaler.transform(row)
    input_tensor = torch.tensor(row_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        class_idx = int(probabilities.argmax())

    one_hot = [1 if i == class_idx else 0 for i in range(len(probabilities))]

    return {
        "class_id": class_idx,
        "class_label": CLASS_LABELS.get(class_idx, str(class_idx)),
        "probabilities": probabilities.tolist(),
        # "one_hot": one_hot,
    }
