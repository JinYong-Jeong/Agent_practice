# FastAPI + LangGraph: LLM 없이, LLM 붙여서

Titanic 승객 정보를 ML 또는 DL 모델로 예측하는 교육용 예제입니다.

같은 FastAPI를 사용해 두 단계를 차례로 실습합니다.

| 단계 | 실행 파일 | 입력 | 배울 내용 |
| --- | --- | --- | --- |
| 1. LLM 없이 | `agent.py` | 직접 작성한 JSON | 상태 → 조건부 분기 → FastAPI 호출 |
| 2. LLM 연결 | `agent_llm.py` | 같은 JSON | 같은 분기/API 호출 → 예측 응답을 LLM이 설명 |

```text
공통: JSON 입력 → 코드로 ML/DL 분기 → FastAPI → 예측 API 응답
                                                  ↓
                         ┌────────────────────────┴─────────────┐
                      1단계                                   2단계
                         ↓                                     ↓
                    format_result                         explain_result
                         ↓                                     ↓
                  정해진 형식으로 출력                    API 응답 → LLM 설명
```

두 버전 모두 사용자가 JSON의 `model`로 ML/DL을 선택하고, 코드의 조건문이 분기합니다. 생존 예측은 FastAPI 뒤의 XGBoost/PyTorch 모델이 수행합니다. **2단계에서는 그 응답을 LLM에 전달해 설명을 생성합니다.** 입력 누락은 두 버전 모두 API 호출 전에 안내합니다.

## 구조

```text
Agent_practice/
├─ agent.py              # 1단계: JSON 입력, LangGraph ML/DL 분기
├─ agent_llm.py          # 2단계: 1단계 분기/API 재사용 + LLM으로 결과 설명
├─ app/
│  ├─ main.py            # FastAPI 엔드포인트
│  ├─ schemas.py         # 요청/응답 형식
│  └─ predictor.py       # 모델 로드와 예측
├─ train/
│  ├─ titanic.csv        # 학습 데이터
│  ├─ dataready.py       # 전처리
│  ├─ mltrain.py         # xgb_model.pkl 생성
│  └─ dltrain.py         # dl_model.pth, scaler.pkl 생성
├─ xgb_model.pkl
├─ dl_model.pth
├─ scaler.pkl
└─ requirements.txt
```

## 설치

Python 3.11, Windows PowerShell 기준입니다. 저장소를 둘 상위 폴더에서 아래 명령을 실행합니다. 이미 clone했다면 `Agent_practice` 폴더로 이동한 뒤 환경 설정부터 진행하세요. 서버와 에이전트 터미널에서 같은 Python 환경을 사용하세요.

```powershell
git clone https://github.com/JinYong-Jeong/Agent_practice.git
cd Agent_practice
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

이미 `fedops311`에 의존성을 설치했다면, 각 터미널에서 `conda activate fedops311` 후 실행하면 됩니다. 1단계에는 LLM API 키가 필요하지 않습니다.

## 모델 학습

저장된 모델 파일이 포함되어 있어 바로 실행할 수 있습니다. 직접 다시 만들려면 저장소 루트에서 실행하세요. 기존 모델 파일을 덮어쓰며, 학습 후에는 FastAPI 서버를 재시작합니다.

```powershell
python -m train.mltrain
python -m train.dltrain
```

```text
train/titanic.csv
   ├─ mltrain.py → xgb_model.pkl
   └─ dltrain.py → dl_model.pth + scaler.pkl
```

`xgb_model.pkl`은 XGBoost 모델, `scaler.pkl`은 DL 입력의 표준화 전처리기입니다. DL 모델은 전체 객체를 pickle로 저장하지 않고 PyTorch 가중치(`state_dict`)를 `dl_model.pth`에 저장하며, 모델 구조는 `train/dltrain.py`의 `BasicMLP`에 있습니다.

## FastAPI 실행

반드시 `Agent_practice` 디렉터리에서 실행합니다.

```powershell
python -m uvicorn app.main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/health>

`No module named 'app'`이 나오면 현재 위치가 `Agent_practice`인지 확인합니다.

## Postman 요청

FastAPI 서버를 먼저 실행한 상태에서 Postman으로 요청합니다.
아래 요청은 예측 API를 직접 호출합니다. LangGraph의 분기는 뒤의 `agent.py`, `agent_llm.py` 실행에서 확인합니다.

### 입력할 수 있는 값

| 필드 | 의미 | 허용값 | 예시 |
| --- | --- | --- | --- |
| `pclass` | 객실 등급 | `1`, `2`, `3` | `1` |
| `sex` | 성별 | `"male"`, `"female"` | `"female"` |
| `fare` | 티켓 요금 | 0 이상의 숫자 | `72` |
| `embarked` | 승선 항구 | `"C"`, `"Q"`, `"S"` | `"C"` |

승선 항구 코드는 다음 의미입니다.

- `C`: Cherbourg
- `Q`: Queenstown
- `S`: Southampton

`model`은 Body에 넣지 않습니다. ML과 DL 중 어떤 모델을 사용할지는 요청 URL로 선택합니다.

### 1. 서버 상태 확인

- Method: `GET`
- URL: `http://127.0.0.1:8000/health`
- Body: 없음

정상 응답:

```json
{
  "status": "ok"
}
```

`/health`는 서버 실행 여부만 확인합니다. 모델은 첫 예측 요청에서 로드하므로 모델 파일까지 정상인지 확인하려면 아래 ML/DL 요청을 각각 보내세요.

### 2. 머신러닝 모델에 예측 요청

- Method: `POST`
- URL: `http://127.0.0.1:8000/predict/ml`
- Body: `raw` → `JSON`

```text
POST http://127.0.0.1:8000/predict/ml
```

### 3. 딥러닝 모델에 예측 요청

- Method: `POST`
- URL: `http://127.0.0.1:8000/predict/dl`
- Body: `raw` → `JSON`

```text
POST http://127.0.0.1:8000/predict/dl
```

ML과 DL은 같은 Body 형식을 사용합니다.

### 복사해서 사용할 수 있는 요청 예시

#### 1등실 여성, Cherbourg 승선

```json
{
  "pclass": 1,
  "sex": "female",
  "fare": 72,
  "embarked": "C"
}
```

#### 3등실 남성, Southampton 승선

```json
{
  "pclass": 3,
  "sex": "male",
  "fare": 7.25,
  "embarked": "S"
}
```

#### 2등실 여성, Queenstown 승선

```json
{
  "pclass": 2,
  "sex": "female",
  "fare": 30,
  "embarked": "Q"
}
```

#### 1등실 남성, Southampton 승선

```json
{
  "pclass": 1,
  "sex": "male",
  "fare": 100,
  "embarked": "S"
}
```

같은 Body를 `/predict/ml`과 `/predict/dl`에 각각 보내면 두 모델의 결과를 비교할 수 있습니다.

### 응답 읽는 방법

```json
{
  "model": "ml",
  "prediction": 1,
  "label": "생존",
  "probabilities": [0.0367, 0.9633]
}
```

- `model`: 사용한 모델. `ml` 또는 `dl`
- `prediction`: 최종 예측. `0`은 사망, `1`은 생존
- `label`: 예측 결과의 한글 표시
- `probabilities[0]`: 사망 확률
- `probabilities[1]`: 생존 확률

### 입력 검증을 확인하는 오류 요청

아래 요청은 의도적으로 `422 Unprocessable Entity`를 발생시킵니다.

#### 필수값 누락

```json
{
  "pclass": 1,
  "sex": "female",
  "fare": 72
}
```

#### 허용되지 않는 객실 등급

```json
{
  "pclass": 4,
  "sex": "female",
  "fare": 72,
  "embarked": "C"
}
```

#### 허용되지 않는 성별 표현

```json
{
  "pclass": 1,
  "sex": "여성",
  "fare": 72,
  "embarked": "C"
}
```

#### 음수 요금과 잘못된 항구 코드

```json
{
  "pclass": 1,
  "sex": "female",
  "fare": -1,
  "embarked": "A"
}
```

이 오류 응답을 통해 `app/schemas.py`의 Pydantic 검증이 어떻게 적용되는지 확인할 수 있습니다.

## 1단계: LLM 없이 LangGraph 실행

FastAPI 서버를 켜둔 상태에서 다른 터미널을 열고 `Agent_practice` 폴더로 이동합니다. 설치 때 만든 환경을 다시 활성화한 뒤 실행합니다. venv를 사용했다면 `.\.venv\Scripts\Activate.ps1`, Conda를 사용했다면 `conda activate fedops311`입니다.

```powershell
python agent.py
```

ML 분기 입력:

```json
{"model": "ml", "pclass": 1, "sex": "female", "fare": 72, "embarked": "C"}
```

DL 분기 입력:

```json
{"model": "dl", "pclass": 1, "sex": "female", "fare": 72, "embarked": "C"}
```

누락값 분기 입력:

```json
{"model": "ml", "pclass": 1}
```

입력 한 번마다 프로그램이 종료됩니다. 다른 예제를 시도할 때는 `python agent.py`를 다시 실행합니다.

ML 입력의 출력 예시:

```text
[check_input]
[call_ml_api]
[format_result]
ML 모델 예측: 생존 (생존 확률 96.3%)
```

`model`을 `dl`로 바꾸면 `[call_dl_api]`를 거칩니다. 필수값이 빠지면 `[ask_for_missing_input]`, `model`이 `ml/dl` 이외의 값이면 `[ask_for_valid_model]`에서 종료됩니다. 확률은 학습 결과에 따라 달라질 수 있습니다.

코드는 이 순서로 읽어보세요.

1. `AgentState`: 노드 사이에 전달되는 값
2. `check_input`: 필수값 확인
3. `route_after_check`와 `add_conditional_edges`: 다음 노드 선택
4. `call_ml_api` / `call_dl_api`: `requests.post()`로 FastAPI 호출
5. `format_result`: API의 예측값을 문장으로 출력

## 2단계: 예측 API의 응답을 LLM이 활용하기

입력 검사와 ML/DL API 호출은 `agent.py`의 함수를 재사용합니다. 마지막 출력 노드만 `format_result`에서 `explain_result`로 바꿉니다. `explain_result`가 **실제 예측 API 응답**을 OpenRouter에 보내 설명을 생성합니다. 로컬 LLM 다운로드나 실행은 필요하지 않습니다.

### OpenRouter 설정

[OpenRouter API 키 페이지](https://openrouter.ai/settings/keys)에서 키를 발급하고, `agent_llm.py`를 실행할 PowerShell에서 환경변수로 설정합니다.

```powershell
$env:OPENROUTER_API_KEY = "발급받은_API_키"
$env:OPENROUTER_MODEL = "poolside/laguna-s-2.1:free"
```

기본값은 [poolside/laguna-s-2.1:free](https://openrouter.ai/poolside/laguna-s-2.1:free)입니다. 로컬 모델 다운로드나 별도 OpenAI API 키는 필요하지 않습니다. `OPENROUTER_MODEL`을 생략해도 이 기본값을 사용합니다.

**무료 모델만 호출합니다.** 특정 모델을 사용하려면 [OpenRouter 모델 목록](https://openrouter.ai/models)에서 `:free`로 끝나는 모델 ID를 선택하세요. 코드는 `openrouter/free` 또는 `:free` 모델만 허용하며, 유료 모델 ID를 설정하면 호출 전에 차단합니다. 무료 모델이 실패해도 유료 모델로 전환하지 않습니다.

이 설정은 설명용 LLM만 바꾸며, 입력 JSON의 `model: "ml"` 또는 `"dl"`과는 별개입니다.

환경변수는 설정한 PowerShell 창에 적용됩니다. 새 창을 열면 다시 설정하세요. 코드나 README에 실제 API 키를 저장하지 않습니다. Python 패키지 추가 설치도 필요 없습니다.

2단계에서 연결하는 곳은 다음과 같습니다.

| 대상 | 주소 | 역할 |
| --- | --- | --- |
| FastAPI | `http://127.0.0.1:8000` | ML/DL 모델로 생존 예측 |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | 예측 응답을 받아 LLM 설명 생성 |

### 실행

FastAPI 서버를 켜두고, API 키를 설정한 에이전트용 터미널에서 실행합니다.

```powershell
python agent_llm.py
```

1단계와 동일한 JSON을 입력합니다. ML 요청:

```json
{"model": "ml", "pclass": 1, "sex": "female", "fare": 72, "embarked": "C"}
```

DL 요청:

```json
{"model": "dl", "pclass": 3, "sex": "male", "fare": 7.25, "embarked": "S"}
```

누락값 요청:

```json
{"model": "ml", "pclass": 1}
```

ML 요청의 출력 예시입니다. 아래 설명은 예시이며, 실제 문장과 예측 확률은 달라질 수 있습니다.

```text
[check_input]
[call_ml_api]
예측 API 응답: {"model": "ml", "prediction": 1, "label": "생존", "probabilities": [0.0367, 0.9633]}
[explain_result]
ML 모델은 생존으로 예측했습니다.
모델이 산출한 생존 확률은 약 96.3%이며, 이는 실제 생존 여부를 확정하는 값은 아닙니다.
```

터미널의 **예측 API 응답**과 LLM 설명을 비교해보세요. LLM에 전달되는 데이터는 다음 두 가지입니다.

- `prediction`: FastAPI가 반환한 모델명, 예측값, 확률 JSON
- `summary`: 코드가 계산한 결과와 생존 확률 문장

프롬프트에서는 API의 수치를 유지하고, 응답에 없는 특성 중요도나 예측 원인을 추측하지 않도록 지시합니다. 이 프롬프트는 설명 생성을 위한 지시이며, 생성 문장의 정확성을 자동으로 보장하는 검증기는 아닙니다.

입력이 누락되거나 `model`이 잘못되면 예측 API와 LLM을 모두 호출하지 않고 안내합니다.

두 버전 모두 입력 한 번을 처리합니다. 이어지는 대화의 정보를 기억하거나 누락값을 누적하지 않으므로, 재실행 시 전체 조건을 다시 입력합니다.

### LLM이 붙은 부분 확인

`agent_llm.py`에서 LLM이 붙는 위치는 예측 API 노드 다음입니다.

```python
graph.add_node("explain_result", explain_result)
graph.add_edge("call_ml_api", "explain_result")
graph.add_edge("call_dl_api", "explain_result")
graph.add_edge("explain_result", END)
```

`explain_result`는 `state["prediction"]`에 저장된 API 응답을 `messages`에 담아 [OpenRouter Chat Completions API](https://openrouter.ai/docs/quickstart)를 호출합니다. 인증은 `Authorization: Bearer ...` 헤더로 전달하며, 생성된 설명은 응답의 `choices[0].message.content`에서 읽습니다.

두 파일의 `build_graph()`를 비교하면, 같은 API 응답을 1단계는 정형 문장으로 출력하고 2단계는 LLM의 설명에 활용한다는 차이를 확인할 수 있습니다.

### 오류를 보는 순서

| 출력 | 확인할 곳 |
| --- | --- |
| `OPENROUTER_API_KEY를 설정하세요` | 실행 중인 PowerShell 창에서 키를 설정했는지 확인 |
| `무료 모델만 허용합니다` | `OPENROUTER_MODEL`을 기본값 또는 실제 존재하는 `:free` 모델 ID로 설정 |
| `OpenRouter 호출 실패` / `401` | API 키가 유효한지 확인 |
| `OpenRouter 호출 실패` / `402` | 계정 잔액이 음수인지, API 키에 사용 제한이 설정되어 있는지 확인. 유료 모델로 전환하지 않음 |
| `OpenRouter 호출 실패` / 그 외 | 모델 ID, 모델 가용성, 네트워크 연결 확인 |
| `LLM 응답 형식 오류` | 빈 응답이나 토큰 제한으로 중단된 응답. 다시 실행하거나 다른 `:free` 모델 선택 |
| `입력이 부족합니다` | 누락 필드를 포함해 JSON을 다시 작성 |
| `API 호출 실패` | FastAPI 실행 여부, HTTP 상태 코드, FastAPI 터미널의 오류 로그 |

예측 API가 실패하면 OpenRouter를 호출하지 않습니다. 예측은 성공했지만 API 키가 없거나 설명 생성이 실패하면, 이미 얻은 예측 결과를 정형 문장으로 출력하고 LLM 오류를 함께 안내합니다.
