"""2단계: JSON → 코드로 ML/DL 분기 → FastAPI 예측 → LLM으로 결과 설명."""

import json
import os

import requests
from langgraph.graph import END, START, StateGraph

from agent import (
    AgentState,
    ask_for_missing_input,
    ask_for_valid_model,
    call_dl_api,
    call_ml_api,
    check_input,
    format_result,
    route_after_check,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "poolside/laguna-s-2.1:free").strip()


SYSTEM_PROMPT = """당신은 Titanic 예측 모델의 응답을 설명하는 도우미입니다.
제공된 예측 API 응답을 한국어로 2~3문장으로 설명하세요.
첫 문장에 모델명(ML 또는 DL), 예측 결과, 생존 확률을 포함하세요.
다음 문장에는 모델의 예측이 실제 생존 여부를 확정하지 않는다는 점을 설명하세요.
prediction의 0은 사망, 1은 생존이며 probabilities는 [사망 확률, 생존 확률]입니다.
summary에는 코드가 계산한 결과와 생존 확률이 있습니다. 이 수치를 그대로 사용하세요.
모델명, 예측 결과, 확률을 바꾸거나 새로운 예측을 만들지 마세요.
제공되지 않은 승객 정보나 예측 원인을 추측하지 마세요.
사고 과정이나 작성 계획은 출력하지 말고 최종 설명만 출력하세요."""


def explain_result(state: AgentState) -> AgentState:
    """예측 API가 반환한 결과를 LLM에 전달해 설명을 생성한다."""

    if state.get("error"):
        return format_result(state)  # API가 실패하면 LLM을 호출하지 않는다.

    summary = format_result(state)["message"]
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        error = "OPENROUTER_API_KEY를 설정하세요. 예측 결과는 정상이며 LLM 설명은 생략합니다."
        return {**state, "error": error, "message": f"{summary}\n{error}"}

    if OPENROUTER_MODEL != "openrouter/free" and not OPENROUTER_MODEL.endswith(":free"):
        error = "무료 모델만 허용합니다. OPENROUTER_MODEL을 openrouter/free 또는 :free로 끝나는 모델 ID로 설정하세요."
        return {**state, "error": error, "message": f"{summary}\n{error}"}

    context = {
        "prediction": state["prediction"],  # 실제 API 응답을 그대로 전달한다.
        "summary": summary,
    }
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
                "stream": False,
                "temperature": 0,
                "max_tokens": 2048,
                "reasoning": {"enabled": False, "exclude": True},
            },
            timeout=(5, 120),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("설명이 토큰 제한으로 중단됐습니다. 다시 실행하거나 다른 무료 모델을 선택하세요.")
        explanation = choice["message"]["content"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("LLM 설명이 비어 있거나 문자열이 아닙니다.")
        return {**state, "message": explanation.strip()}
    except requests.RequestException as exc:
        error = f"OpenRouter 호출 실패: {exc}\nAPI 키, 무료 모델의 가용성 및 요청 한도를 확인하세요."
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        error = f"LLM 응답 형식 오류: {exc}"
    # LLM을 사용할 수 없어도 이미 얻은 예측 결과는 보여준다.
    return {**state, "error": error, "message": f"{summary}\n{error}"}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("check_input", check_input)
    graph.add_node("ask_for_missing_input", ask_for_missing_input)
    graph.add_node("ask_for_valid_model", ask_for_valid_model)
    graph.add_node("call_ml_api", call_ml_api)
    graph.add_node("call_dl_api", call_dl_api)
    graph.add_node("explain_result", explain_result)

    # 입력 검사와 ML/DL 분기는 1단계와 동일하게 코드가 수행한다.
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
    # API 응답을 받은 뒤에만 LLM이 결과를 설명한다.
    graph.add_edge("call_ml_api", "explain_result")
    graph.add_edge("call_dl_api", "explain_result")
    graph.add_edge("explain_result", END)
    return graph.compile()


llm_agent = build_graph()


def main() -> None:
    print(f"JSON을 입력하세요. 예측 결과를 LLM이 설명합니다. (OpenRouter 모델: {OPENROUTER_MODEL})")
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

    for update in llm_agent.stream(user_input, stream_mode="updates"):
        for node_name, result in update.items():
            print(f"[{node_name}]")
            if node_name in ("call_ml_api", "call_dl_api") and not result.get("error"):
                print("예측 API 응답:", json.dumps(result["prediction"], ensure_ascii=False))
    print(result["message"])


if __name__ == "__main__":
    main()
