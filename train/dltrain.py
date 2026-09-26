"""PyTorch 딥러닝 모델을 학습해 모델과 스케일러를 저장한다."""

from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from train.dataready import load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "dl_model.pth"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(inputs))
        hidden = torch.relu(self.fc2(hidden))
        return self.fc3(hidden)


def train_model(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    epochs: int = 100,
) -> BasicMLP:
    torch.manual_seed(42)
    np.random.seed(42)

    features, labels = load_training_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features.to_numpy(dtype=np.float32),
        labels.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    train_inputs = torch.tensor(x_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    valid_inputs = torch.tensor(x_valid, dtype=torch.float32)

    model = BasicMLP(train_inputs.shape[1])
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_inputs), train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(valid_inputs).argmax(dim=1).numpy()
    accuracy = accuracy_score(y_valid, predictions)

    torch.save(model.state_dict(), model_path)
    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    print(f"validation accuracy: {accuracy:.3f}")
    print(f"saved: {model_path}")
    print(f"saved: {scaler_path}")
    return model


if __name__ == "__main__":
    train_model()
