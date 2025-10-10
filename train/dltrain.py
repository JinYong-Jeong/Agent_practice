import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from train.dataready import df_preset

TARGET_COLUMN = "Survived"
DEFAULT_MODEL_PATH = "dl_model.pth"
DEFAULT_SCALER_PATH = "scaler.pkl"


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def prepare_features():
    """Return feature matrix and labels as numpy arrays."""
    features = df_preset.drop(TARGET_COLUMN, axis=1).values
    labels = df_preset[TARGET_COLUMN].values
    return features, labels


def train_model(
    epochs: int = 100,
    learning_rate: float = 0.01,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    verbose: bool = True,
):
    features, labels = prepare_features()

    X_train, X_valid, y_train, y_valid = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

    model = BasicMLP(X_train_tensor.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    if verbose:
        with torch.no_grad():
            logits = model(X_valid_tensor)
            preds = torch.argmax(logits, dim=1)
            onehot_preds = nn.functional.one_hot(preds, num_classes=2)
            print("one_hot predict:")
            print(onehot_preds.numpy())

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    if verbose:
        print("dl.pth saved")
        print("scaler.pkl saved")

    return model, scaler


if __name__ == "__main__":
    train_model()