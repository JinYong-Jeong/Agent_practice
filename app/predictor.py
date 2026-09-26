"""pickle ML 모델과 PyTorch DL 모델을 불러와 예측한다."""

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from app.schemas import PredictionRequest, PredictionResponse
from train.dataready import FEATURE_COLUMNS, encode_passenger
from train.dltrain import BasicMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_MODEL_PATH = PROJECT_ROOT / "xgb_model.pkl"
DL_MODEL_PATH = PROJECT_ROOT / "dl_model.pth"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"

@lru_cache(maxsize=1)
def load_ml_model():
    """pickle 모델은 첫 ML 요청에서 한 번만 로드한다."""

    with ML_MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_dl_artifacts():
    """딥러닝 모델과 전처리기는 첫 DL 요청에서 한 번만 로드한다."""

    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)

    model = BasicMLP(len(FEATURE_COLUMNS))
    state_dict = torch.load(DL_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def encode_features(payload: PredictionRequest) -> dict[str, float]:
    """API 입력을 두 모델이 학습한 컬럼 형태로 변환한다."""

    return encode_passenger(
        pclass=payload.pclass,
        sex=payload.sex,
        fare=payload.fare,
        embarked=payload.embarked,
    )


def _response(model: str, prediction: int, probabilities: list[float]) -> PredictionResponse:
    return PredictionResponse(
        model=model,
        prediction=prediction,
        label="생존" if prediction == 1 else "사망",
        probabilities=probabilities,
    )


def predict_ml(payload: PredictionRequest) -> PredictionResponse:
    model = load_ml_model()
    features = pd.DataFrame([encode_features(payload)], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(features, validate_features=False)[0])
    probabilities = model.predict_proba(features, validate_features=False)[0].tolist()
    return _response("ml", prediction, probabilities)


def predict_dl(payload: PredictionRequest) -> PredictionResponse:
    model, scaler = load_dl_artifacts()
    feature_dict = encode_features(payload)
    row = np.array([[feature_dict[name] for name in FEATURE_COLUMNS]], dtype=np.float32)
    scaled = scaler.transform(row)
    inputs = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).numpy()[0].tolist()

    prediction = int(np.argmax(probabilities))
    return _response("dl", prediction, probabilities)
