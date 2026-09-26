"""로컬 Titanic CSV를 모델 학습용 데이터로 변환한다."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).with_name("titanic.csv")
TARGET_COLUMN = "survived"
FEATURE_COLUMNS = [
    "Fare",
    "Pclass_2",
    "Pclass_3",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]


def encode_passenger(
    pclass: int,
    sex: str,
    fare: float,
    embarked: str,
) -> dict[str, float]:
    return {
        "Fare": float(fare),
        "Pclass_2": float(pclass == 2),
        "Pclass_3": float(pclass == 3),
        "Sex_male": float(sex == "male"),
        "Embarked_Q": float(embarked == "Q"),
        "Embarked_S": float(embarked == "S"),
    }


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """API와 동일한 전처리를 사용해 특성과 정답을 반환한다."""

    data = pd.read_csv(DATA_PATH)
    data = data[[TARGET_COLUMN, "pclass", "sex", "fare", "embarked"]].dropna()

    encoded_rows = [
        encode_passenger(
            pclass=row.pclass,
            sex=row.sex,
            fare=row.fare,
            embarked=row.embarked,
        )
        for row in data.itertuples(index=False)
    ]
    features = pd.DataFrame(encoded_rows, columns=FEATURE_COLUMNS)
    labels = data[TARGET_COLUMN].astype(int).reset_index(drop=True)
    return features, labels


if __name__ == "__main__":
    x_data, y_data = load_training_data()
    print(f"samples: {len(x_data)}")
    print(f"features: {x_data.columns.tolist()}")
    print(f"label counts:\n{y_data.value_counts().sort_index()}")
