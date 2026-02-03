# train_bc.py

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ===================== CONFIG =====================
CSV_PATH = "dataset.csv"
BATCH_SIZE = 512
EPOCHS = 600
LR = 1e-3

MODEL_PATH = "trojan_model.pt"
X_SCALER_PATH = "x_scaler_trojan_model.pkl"
Y_SCALER_PATH = "y_scaler_trojan_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10  # fraction of train split

# ===================== DEVICE =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


# ===================== DATASET =====================
class WheelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ===================== MODEL =====================
class Policy(nn.Module):
    def __init__(self, in_dim: int = 5, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    # -------- Load data --------
    df = pd.read_csv(CSV_PATH)

    x_cols = ["x", "y", "theta", "x_ref", "y_ref"]
    y_cols = ["multiplier"]

    X_raw = df[x_cols].to_numpy()
    y_raw = df[y_cols].to_numpy()

    # -------- Split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # -------- Normalize --------
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)  # keep scaled for loader; invert later

    # -------- Loaders --------
    train_loader = DataLoader(
        WheelDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WheelDataset(X_val, y_val),
        batch_size=1024,
        shuffle=False,
    )
    test_loader = DataLoader(
        WheelDataset(X_test, y_test_scaled),
        batch_size=1024,
        shuffle=False,
    )

    # -------- Model / Optim --------
    model = Policy(in_dim=len(x_cols), out_dim=len(y_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # -------- Train --------
    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # -------- Validate --------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: best_state was never set.")

    # -------- Test (in original scale) --------
    model.load_state_dict(best_state)
    model.eval()

    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            preds_scaled.append(model(xb).cpu().numpy())

    y_pred_scaled = np.vstack(preds_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_test  # original scale

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    print(f"test RMSE: {rmse:.6f}")
    print(f"test MAE : {mae:.6f}")

    # -------- Save artifacts --------
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    print("saved:")
    print(" -", MODEL_PATH)
    print(" -", X_SCALER_PATH)
    print(" -", Y_SCALER_PATH)


if __name__ == "__main__":
    main()
