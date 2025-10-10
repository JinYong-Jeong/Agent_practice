from __future__ import annotations
import streamlit as st

import json
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

from app.services.agent_service import predict_by_model

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

ModelType = Literal["dl", "ml"]
FEATURE_FIELDS = ("pclass", "sex", "fare", "embarked")
EMBARKED_GLOSSARY = "(C=Cherbourg, Q=Queenstown, S=Southampton)"

llm = ChatOllama(model="gpt-oss:20b", base_url="")


class ParsedInputs(BaseModel):
    model: Optional[ModelType] = Field(None, description="Choose either dl or ml")
    pclass: Optional[int] = Field(None, description="Passenger class 1, 2, or 3")
    sex: Optional[str] = Field(None, description="Passenger sex: male or female")
    fare: Optional[float] = Field(None, description="Ticket fare (numeric)")
    embarked: Optional[str] = Field(None, description="Embarkation port: C, Q, or S")

    def normalise(self) -> "ParsedInputs":
        if self.sex:
            self.sex = self.sex.strip().lower()
        if self.embarked:
            self.embarked = self.embarked.strip().upper()
        if self.model:
            self.model = self.model.strip().lower()  # type: ignore[assignment]
        return self

    def missing_fields(self) -> List[str]:
        missing: List[str] = []
        if self.model not in ("dl", "ml"):
            missing.append("model")
        for field_name in FEATURE_FIELDS:
            if getattr(self, field_name) in (None, ""):
                missing.append(field_name)
        return missing

    def to_features(self) -> Dict[str, Any]:
        return {
            "pclass": self.pclass,
            "sex": self.sex,
            "fare": self.fare,
            "embarked": self.embarked,
        }

# 상태
class AgentState(TypedDict, total=False):
    question: str
    parsed: ParsedInputs
    missing_fields: List[str]
    prediction: Dict[str, Any]
    model_type: ModelType
    response: str


class TitanicAgent:
    def __init__(self, *, llm: Runnable) -> None:
        self.llm = llm
        self._build_parsers()
        self._build_graph()

    def chat(self, question: str) -> str:
        final_state = self._app.invoke({"question": question})
        response = final_state.get("response")
        if not response:
            raise RuntimeError("Agent produced an empty response")
        return response

    def _build_parsers(self) -> None:
        parser = JsonOutputParser(pydantic_object=ParsedInputs)

        parse_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 Titanic 예측을 위한 정보 추출 도우미입니다."
                    " 사용자의 문장에서 dl 또는 ml 중 어떤 모델을 원하는지,"
                    " pclass(1/2/3), sex(male/female), fare(숫자), embarked(C/Q/S) 값을 JSON으로 추출하세요."
                    " 모르면 null로 두세요. {glossary}",
                ),
                ("human", "질문: {question}\n{format_instructions}"),
            ]
        )

        respond_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 Titanic 생존 예측 비서입니다."
                    " 모델 선택과 예측 결과를 한국어로 요약하고, label 0=사망, 1=생존임을 항상 알려주세요."
                    f" 승선 항구 코드 설명은 {EMBARKED_GLOSSARY} 입니다.",
                ),
                (
                    "human",
                    "질문: {question}\n모델: {model}\n입력 특성: {features}\n예측 결과: {prediction}",
                ),
            ]
        )

        self._parse_chain: Runnable = parse_prompt | self.llm | parser
        self._respond_chain: Runnable = respond_prompt | self.llm
        self._parse_instructions = parser.get_format_instructions()

    def _build_graph(self) -> None:
        graph = StateGraph(AgentState)
        graph.add_node("parse", self._node_parse)
        graph.add_node("missing", self._node_missing)
        graph.add_node("predict_dl", self._node_predict_dl)
        graph.add_node("predict_ml", self._node_predict_ml)
        graph.add_node("respond", self._node_respond)

        graph.set_entry_point("parse")
        graph.add_conditional_edges(
            "parse",
            self._route_after_parse,
            {
                "missing": "missing",
                "dl": "predict_dl",
                "ml": "predict_ml",
            },
        )
        graph.add_edge("missing", END)
        graph.add_edge("predict_dl", "respond")
        graph.add_edge("predict_ml", "respond")
        graph.add_edge("respond", END)

        self._app = graph.compile()

    def _node_parse(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        try:
            raw = self._parse_chain.invoke(
                {
                    "question": question,
                    "format_instructions": self._parse_instructions,
                    "glossary": EMBARKED_GLOSSARY,
                }
            )
        except ValidationError as exc:
            logger.warning("Failed to parse question: %s", exc)
            parsed = ParsedInputs()
        else:
            if isinstance(raw, ParsedInputs):
                parsed = raw
            elif isinstance(raw, dict):
                try:
                    parsed = ParsedInputs(**raw)
                except ValidationError as exc:
                    logger.warning("Failed to coerce parsed dict: %s", exc)
                    parsed = ParsedInputs()
            else:
                logger.warning("Unexpected parser output type: %s", type(raw))
                parsed = ParsedInputs()

        parsed = parsed.normalise()
        missing = parsed.missing_fields()
        new_state: AgentState = {
            "question": question,
            "parsed": parsed,
            "missing_fields": missing,
        }
        if parsed.model in ("dl", "ml"):
            new_state["model_type"] = parsed.model  # type: ignore[assignment]
        return new_state

    def _node_missing(self, state: AgentState) -> AgentState:
        message = (
            "다음 정보를 모두 입력해주세요: 모델(dl 또는 ml), "
            "객실 등급(pclass), 성별(sex), 요금(fare), 승선 항구(embarked) "
            f"{EMBARKED_GLOSSARY}."
        )
        return {**state, "response": message}

    def _node_predict_dl(self, state: AgentState) -> AgentState:
        return self._predict_with_model(state, "dl")

    def _node_predict_ml(self, state: AgentState) -> AgentState:
        return self._predict_with_model(state, "ml")

    def _predict_with_model(self, state: AgentState, model_type: ModelType) -> AgentState:
        parsed: ParsedInputs = state["parsed"]
        features = parsed.to_features()
        try:
            result = predict_by_model(
                pclass=int(features["pclass"]),
                sex=str(features["sex"]),
                fare=float(features["fare"]),
                embarked=str(features["embarked"]),
                model_type=model_type,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("입력 값이 충분하지 않습니다.") from exc
        payload = result.get("result")
        if isinstance(payload, dict):
            payload["label_mapping_ko"] = {0: "사망", 1: "생존"}
        return {**state, "prediction": result, "model_type": model_type}

    def _node_respond(self, state: AgentState) -> AgentState:
        prediction = state["prediction"]
        features_json = json.dumps(prediction["features"], ensure_ascii=False)
        prediction_json = json.dumps(prediction["result"], ensure_ascii=False)
        response = self._respond_chain.invoke(
            {
                "question": state.get("question", ""),
                "model": state.get("model_type", "dl"),
                "features": features_json,
                "prediction": prediction_json,
            }
        )
        text = response.content if hasattr(response, "content") else str(response)
        return {**state, "response": text, "missing_fields": []}

    def _route_after_parse(self, state: AgentState) -> str:
        if state.get("missing_fields"):
            return "missing"
        model_type: Optional[ModelType] = state.get("model_type")  # type: ignore[assignment]
        return model_type or "missing"


def _interactive_chat(*, llm: Runnable) -> None:
    agent = TitanicAgent(llm=llm)
    print("모델 종류, 객실 등급, 성별, 요금, 승선 항구를 입력하여 생존여부를 확인하세요. 종료하려면 quit 입력.")
    while True:
        question = input("You> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        try:
            answer = agent.chat(question)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("에이전트 오류: %s", exc)
            continue
        print(f"Agent> {answer}\n")



# ========== Streamlit UI ==========

st.title("타이타닉 생존 예측 에이전트")
# 최초 1회 그래프 준비
if "graph" not in st.session_state:
    from your_module import app  # StateGraph.compile() 한 객체
    st.session_state.graph = app
if "state" not in st.session_state:
    st.session_state.state = {}

st.title("Titanic Agent")

# 과거 대화 출력
for role, msg in st.session_state.get("messages", []):
    with st.chat_message(role):
        st.markdown(msg)

# 사용자 입력
q = st.chat_input("자연어로 질문하세요")
if q:
    # 그래프 실행
    st.session_state.state["question"] = q
    final_state = st.session_state.graph.invoke(st.session_state.state)

    # 출력·상태 저장
    st.session_state.state = final_state
    st.session_state.messages = st.session_state.get("messages", []) + [("user", q), ("assistant", final_state.get("response",""))]

    with st.chat_message("assistant"):
        st.markdown(final_state.get("response", "(응답 없음)"))


