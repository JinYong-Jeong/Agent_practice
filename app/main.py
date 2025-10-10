from fastapi import FastAPI, HTTPException
import uvicorn
from app.schemas import (
    AgentPredictionRequest,
    AgentPredictionResponse,
    PredictionRequest,
    DLPredictionResponse,
    MLPredictionResponse,
)
# from app.services.agent_service import predict_by_model
from app.services.dl_service import encode_features, predict_from_dict
from app.services.ml_service import predict_from_features

app = FastAPI(title="Titanic Prediction API", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/dl", response_model=DLPredictionResponse)
def predict_dl(payload: PredictionRequest):
    feature_dict = encode_features(payload.pclass, payload.sex, payload.fare, payload.embarked)

    try:
        result = predict_from_dict(feature_dict)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/predict/ml", response_model=MLPredictionResponse)
def predict_ml(payload: PredictionRequest):
    feature_dict = encode_features(payload.pclass, payload.sex, payload.fare, payload.embarked)

    try:
        result = predict_from_features(feature_dict)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


# @app.post("/predict/agent", response_model=AgentPredictionResponse)
# def predict_agent(payload: AgentPredictionRequest):
#     try:
#         result = predict_by_model(
#             payload.pclass,
#             payload.sex,
#             payload.fare,
#             payload.embarked,
#             payload.model_type,
#         )
#     except FileNotFoundError as exc:
#         raise HTTPException(status_code=500, detail=str(exc)) from exc

#     return result

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
