"""Titanic 예측 FastAPI 서버."""

from fastapi import FastAPI

from app.predictor import predict_dl, predict_ml
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Titanic ML/DL Prediction API",
    description="FastAPI, Postman, LangGraph 분기를 배우기 위한 최소 예제",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """서버가 실행 중인지 확인한다."""

    return {"status": "ok"}


@app.post("/predict/ml", response_model=PredictionResponse)
def predict_with_ml(payload: PredictionRequest) -> PredictionResponse:
    """pickle로 저장된 XGBoost 모델로 예측한다."""

    return predict_ml(payload)


@app.post("/predict/dl", response_model=PredictionResponse)
def predict_with_dl(payload: PredictionRequest) -> PredictionResponse:
    """PyTorch 딥러닝 모델로 예측한다."""

    return predict_dl(payload)
