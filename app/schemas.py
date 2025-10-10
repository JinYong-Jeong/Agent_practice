from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, or 3)")
    sex: Literal["male", "female"]
    fare: float = Field(..., ge=0, description="Ticket fare")
    embarked: Literal["C", "Q", "S"]


class DLPredictionResponse(BaseModel):
    class_id: int
    class_label: str
    # one_hot: list[int]
    probabilities: list[float]


class MLPredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float] = Field(default_factory=list)


class AgentPredictionRequest(PredictionRequest):
    model_type: Literal["dl", "ml"] = Field(
        ..., description="Select which model to run (dl or ml)"
    )


class AgentPredictionResponse(BaseModel):
    model_type: Literal["dl", "ml"]
    features: dict[str, float]
    result: DLPredictionResponse | MLPredictionResponse