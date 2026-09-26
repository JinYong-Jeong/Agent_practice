"""API 요청과 응답의 데이터 형식."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="객실 등급: 1, 2, 3")
    sex: Literal["male", "female"]
    fare: float = Field(..., ge=0, description="티켓 요금")
    embarked: Literal["C", "Q", "S"]


class PredictionResponse(BaseModel):
    model: Literal["ml", "dl"]
    prediction: Literal[0, 1]
    label: Literal["사망", "생존"]
    probabilities: list[float] = Field(..., min_length=2, max_length=2)
