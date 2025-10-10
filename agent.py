from __future__ import annotations
import streamlit as st

## 사용자한테서 입력받은 정보를 json 형태로 파싱해야해서 import함
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.agents import Tool
from typing import TypedDict,Any,Literal
import requests
import json
from pydantic import BaseModel, Field, ValidationError


# ========== 설정 ==========
llm = ChatOllama(model="gpt-oss:20b", base_url="")

# ========== 용어 사전 ========== 
ModelType = Literal["dl", "ml"]
FEATURE_FIELDS = ("pclass", "sex", "fare", "embarked")
EMBARKED_GLOSSARY = "(C=Cherbourg, Q=Queenstown, S=Southampton)"


# ==============상태================
class AgentState(TypedDict, total=False):
    question: str
    missing_fields: list[str]
    prediction: dict[str, Any]
    response: str
    model_type: ModelType
    parsed: ParsedInputs
    features: dict[str, Any]

# ========= 입력 스키마 ==========
class ParsedInputs(BaseModel):
    model: ModelType | None = Field(None, description="dl or ml")
    pclass: int | None = Field(None, description="1/2/3")
    sex: str | None = Field(None, description="male/female")
    fare: float | None = Field(None, description="numeric")
    embarked: str | None = Field(None, description="C/Q/S")

    def normalise(self) -> "ParsedInputs":
        if self.sex: self.sex = self.sex.strip().lower()
        if self.embarked: self.embarked = self.embarked.strip().upper()
        if self.model: self.model = self.model.strip().lower()  # type: ignore[assignment]
        return self

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if self.model not in ("dl","ml"): missing.append("model")
        for f in FEATURE_FIELDS:
            if getattr(self, f) in (None, ""): missing.append(f)
        return missing

    def to_features(self) -> dict[str, Any]:
        return {"pclass": self.pclass, "sex": self.sex, "fare": self.fare, "embarked": self.embarked}


# ========== 기능 함수 ==========

# === 파씽 과정 ===
parser = JsonOutputParser(pydantic_object=ParsedInputs)
parse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "사용자 문장에서 dl/ml, pclass(1/2/3), sex(male/female), "
     "fare(숫자), embarked(C/Q/S)를 JSON으로 추출. 모르면 null. {glossary}"),
    ("human", "질문: {question}\n{format_instructions}")
])
parse_chain = parse_prompt | llm | parser
fmt = parser.get_format_instructions()



# === 프롬프트 ===
respond_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 Titanic 생존 예측 비서입니다. "
     "모델 선택과 예측 결과를 한국어로 요약하고, label 0=사망, 1=생존임을 항상 알려주세요. "
     f"승선 항구 코드 설명은 {EMBARKED_GLOSSARY} 입니다."
    ),
    ("human",
     "질문: {question}\n"
     "모델: {model}\n"
     "입력 특성: {features}\n"
     "예측 결과: {prediction}"
    ),
])
respond_chain = respond_prompt | llm

# === API 헬퍼 ===
def predict_by_model(features: dict[str, Any], model_type: ModelType) -> dict[str, Any]:
    print("입력값", features)
    if model_type == "dl":
        print("DL 모델 호출")
        prediction = requests.post("http://127.0.0.1:8000/predict/dl", json=features)
    elif model_type == "ml":
        print("ML 모델 호출")
        prediction = requests.post("http://127.0.0.1:8000/predict/ml", json=features)
    else:
        raise ValueError("model_type은 'dl' or 'ml'중 골라야합니다.")
    print("출력값", prediction.json())
    return {"result": prediction.json()}  # 유지 (아래 node에서 평탄화)

# === 노드 함수===
def node_missing(state: AgentState) -> AgentState:
    msg = (
        "다음 정보를 모두 입력해주세요: 모델(dl 또는 ml), "
        "객실 등급(pclass), 성별(sex), 요금(fare), 승선 항구(embarked) "
        f"{EMBARKED_GLOSSARY}."
    )
    return {**state, "response": msg}

# 1) 입력값 파씽
def node_parse(state: AgentState) -> AgentState:
    raw = parse_chain.invoke({
        "question": state.get("question",""),
        "format_instructions": fmt,
        "glossary": EMBARKED_GLOSSARY,
    })
    parsed = raw if isinstance(raw, ParsedInputs) else ParsedInputs(**raw)
    return {**state, "parsed": parsed.normalise()}

# 2) 결측 필드 체크
def node_check_missing(state: AgentState) -> AgentState:
    missing_list = state["parsed"].missing_fields()
    return {**state, "missing_fields": missing_list}

# 3) 라우팅(분기 결정)
def route_after_parse(state: AgentState) -> str:
    if state.get("missing_fields"):
        return "missing"
    m = state["parsed"].model
    if m == "dl": return "dl"
    if m == "ml": return "ml"
    return "missing"

# 4) 특성 만들기
def node_to_features(state: AgentState) -> AgentState:
    feats: dict[str, Any] = state["parsed"].to_features()
    mt = state["parsed"].model
    return {**state, "features": feats, "model_type": mt}

# 5) API 호출 (헬퍼 사용 + 예측만 평탄화)
def node_call_api(state: AgentState) -> AgentState:
    data = predict_by_model(state["features"], state["model_type"])
    return {**state, "prediction": data["result"]}  # ← {"result": ...} -> ... 으로 평탄화

# 6) 응답 생성
def node_respond(state: AgentState) -> AgentState:
    resp = respond_chain.invoke({
        "question": state.get("question",""),
        "model": state.get("model_type","dl"),
        "features": json.dumps(state["features"], ensure_ascii=False),
        "prediction": json.dumps(state["prediction"], ensure_ascii=False),
    })
    text = getattr(resp, "content", str(resp))
    return {**state, "response": text}


# ========== LangGraph 정의 ==========

graph = StateGraph(AgentState)
graph.add_node("check", node_check_missing)
graph.add_node("missing", node_missing)
graph.add_node("to_features", node_to_features)
graph.add_node("call_api", node_call_api)
graph.add_node("respond", node_respond)
graph.add_node("parse", node_parse)

graph.set_entry_point("parse")
graph.add_edge("parse", "check")
graph.add_conditional_edges("check", route_after_parse, {
    "missing": "missing",
    "dl": "to_features",
    "ml": "to_features",
})
graph.add_edge("to_features", "call_api")
graph.add_edge("call_api", "respond")
graph.add_edge("respond", END)

app = graph.compile()


if __name__ == "__main__":
    state: AgentState = {}
    print("자연어로 물어보세요.(quit로 종료) 예) dl로 예측해줘. 1등실 여성, 요금 72, C에서 탔어")
    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() == "quit":
            break
        state["question"] = q
        state = app.invoke(state)
        print("Agent>", state.get("response", "(응답 없음)"))








