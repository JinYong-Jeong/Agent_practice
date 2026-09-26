"""XGBoost 머신러닝 모델을 학습해 xgb_model.pkl을 만든다."""

from pathlib import Path
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from train.dataready import load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "xgb_model.pkl"


def train_model(output_path: Path = MODEL_PATH) -> XGBClassifier:
    features, labels = load_training_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_valid, model.predict(x_valid))
    with output_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"validation accuracy: {accuracy:.3f}")
    print(f"saved: {output_path}")
    return model


if __name__ == "__main__":
    train_model()
