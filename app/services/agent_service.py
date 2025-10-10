from __future__ import annotations

from typing import Dict, Literal

from .dl_service import encode_features, predict_from_dict
from .ml_service import predict_from_features

ModelType = Literal["dl", "ml"]


def predict_by_model(
    pclass: int,
    sex: str,
    fare: float,
    embarked: str,
    model_type: ModelType,
) -> Dict[str, object]:
    features = encode_features(pclass, sex, fare, embarked)

    if model_type == "dl":
        prediction = predict_from_dict(features)
    else:
        prediction = predict_from_features(features)

    return {
        "model_type": model_type,
        "features": features,
        "result": prediction,
    }