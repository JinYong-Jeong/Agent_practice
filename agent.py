"""1단계: LLM 없이 JSON → LangGraph 분기 → FastAPI 호출.

LLM 없이 LangGraph의 상태, 노드, 조건부 분기와 API 호출에만 집중한다.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

import requests
from langgraph.graph import END, START, StateGraph


API_BASE_URL = "http://127.0.0.1:8000"
REQUIRED_FIELDS = ("model", "pclass", "sex", "fare", "embarked")


class AgentState(TypedDict, total=False):
    """노드 사이를 이동하는 공용 상태."""

    model: str
    pclass: int
    sex: str
    fare: float
    embarked: str
    missing_fields: list[str]
    prediction: dict[str, Any]
    error: str
    message: str


def check_input(state: AgentState) -> AgentState:
    """API 호출에 필요한 값이 모두 있는지 확인한다."""

    missing = [name for name in REQUIRED_FIELDS if state.get(name) in (None, "")]
    return {**state, "missing_fields": missing}


def route_after_check(state: AgentState) -> str:
    """누락값과 모델 선택에 따라 다음 노드를 결정한다."""

    if state["missing_fields"]:
        return "missing"
    if state["model"] == "ml":
        return "ml"
    if state["model"] == "dl":
        return "dl"
    return "invalid_model"


def ask_for_missing_input(state: AgentState) -> AgentState:
    fields = ", ".join(state["missing_fields"])
    return {**state, "message": f"입력이 부족합니다: {fields}"}


def ask_for_valid_model(state: AgentState) -> AgentState:
    return {**state, "message": "model은 ml 또는 dl이어야 합니다."}


def _call_api(state: AgentState, model: str) -> AgentState:
    payload = {name: state[name] for name in REQUIRED_FIELDS if name != "model"}
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/{model}",
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
        return {**state, "prediction": response.json()}
    except requests.RequestException as exc:
        return {**state, "error": f"API 호출 실패: {exc}"}


def call_ml_api(state: AgentState) -> AgentState:
    """머신러닝 예측 API를 호출한다."""

    return _call_api(state, "ml")


def call_dl_api(state: AgentState) -> AgentState:
    """딥러닝 예측 API를 호출한다."""

    return _call_api(state, "dl")


def format_result(state: AgentState) -> AgentState:
    """API 응답을 사람이 읽기 쉬운 문장으로 바꾼다."""

    if state.get("error"):
        return {**state, "message": state["error"]}

    result = state["prediction"]
    probability = result["probabilities"][1] * 100
    message = (
        f"{result['model'].upper()} 모델 예측: {result['label']} "
        f"(생존 확률 {probability:.1f}%)"
    )
    return {**state, "message": message}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("check_input", check_input)
    graph.add_node("ask_for_missing_input", ask_for_missing_input)
    graph.add_node("ask_for_valid_model", ask_for_valid_model)
    graph.add_node("call_ml_api", call_ml_api)
    graph.add_node("call_dl_api", call_dl_api)
    graph.add_node("format_result", format_result)

    graph.add_edge(START, "check_input")
    graph.add_conditional_edges(
        "check_input",
        route_after_check,
        {
            "missing": "ask_for_missing_input",
            "invalid_model": "ask_for_valid_model",
            "ml": "call_ml_api",
            "dl": "call_dl_api",
        },
    )
    graph.add_edge("ask_for_missing_input", END)
    graph.add_edge("ask_for_valid_model", END)
    graph.add_edge("call_ml_api", "format_result")
    graph.add_edge("call_dl_api", "format_result")
    graph.add_edge("format_result", END)
    return graph.compile()


agent = build_graph()


def main() -> None:
    print("JSON을 입력하세요.")
    print('예: {"model": "ml", "pclass": 1, "sex": "female", "fare": 72, "embarked": "C"}')
    raw = input("> ").strip()

    try:
        user_input = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"올바른 JSON이 아닙니다: {exc}")
        return

    if not isinstance(user_input, dict):
        print("JSON 객체를 입력해주세요.")
        return

    # stream은 각 노드가 끝날 때 상태를 반환한다. 실행 순서를 눈으로 확인한다.
    for update in agent.stream(user_input, stream_mode="updates"):
        for node_name, result in update.items():
            print(f"[{node_name}]")
    print(result["message"])


if __name__ == "__main__":
    main()
