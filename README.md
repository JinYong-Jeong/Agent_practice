# Titanic Prediction Agent Practice

자연어로 입력한 승객 정보를 구조화하고, 머신러닝 또는 딥러닝 모델로 Titanic 생존 여부를 예측하는 Python 실습 프로젝트입니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `agent.py` | LangGraph·Ollama 기반 입력 파싱, 누락 정보 확인, 예측 API 호출 |
| `pre_agent.py` | 별도 에이전트 실습 코드 |
| `app/main.py` | FastAPI 예측 서버 |
| `app/services/` | 머신러닝·딥러닝 예측 서비스 |
| `train/` | 데이터 준비와 모델 학습 코드 |
| `dl_model.pth`, `xgb_model.pkl`, `scaler.pkl` | 저장된 모델과 전처리 산출물 |

## 흐름

사용자 입력 → 모델 종류와 승객 특성 파싱 → 누락 정보 확인 → 예측 API 호출 → 한국어 결과 설명.

입력 특성은 `pclass`, `sex`, `fare`, `embarked`입니다. API는 `GET /health`, `POST /predict/dl`, `POST /predict/ml`을 제공합니다.

## 실행 전 확인

현재 저장소에는 의존성 명세 파일이 없습니다. Python 패키지 환경을 준비하고 `agent.py`의 Ollama 주소와 모델 설정을 확인해야 합니다. 에이전트는 로컬 `127.0.0.1:8000` 예측 서버를 호출합니다.

학습용 코드이며, 이 문서 추가 과정에서 모델 재학습이나 전체 실행 검증은 수행하지 않았습니다.
