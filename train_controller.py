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
EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-5

TEST_SIZE = 0.10
VAL_SIZE = 0.10
SEED = 42

MODEL_PATH = "controller_model.pt"
X_SCALER_PATH = "x_scaler_controller_model.pkl"
Y_SCALER_PATH = "y_scaler_controller_model.pkl"

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
    def __init__(self, in_dim: int = 5, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    # -------- Load data --------
    df = pd.read_csv(CSV_PATH)

    x_cols = ["x", "y", "theta", "x_ref", "y_ref"]
    y_cols = ["left_cmd", "right_cmd"]

    X = df[x_cols].values
    y = df[y_cols].values

    # -------- Split --------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=VAL_SIZE, random_state=SEED
    )

    # -------- Scale --------
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr)

    X_tr = x_scaler.transform(X_tr)
    X_va = x_scaler.transform(X_va)
    X_te = x_scaler.transform(X_te)

    y_tr = y_scaler.transform(y_tr)
    y_va = y_scaler.transform(y_va)
    y_te_s = y_scaler.transform(y_te)

    # -------- Loaders --------
    train_loader = DataLoader(
        WheelDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WheelDataset(X_va, y_va),
        batch_size=1024,
        shuffle=False,
    )
    test_loader = DataLoader(
        WheelDataset(X_te, y_te_s),
        batch_size=1024,
        shuffle=False,
    )

    # -------- Train --------
    model = Policy(in_dim=len(x_cols), out_dim=len(y_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

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

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item())

        tr_loss = float(np.mean(train_losses))
        va_loss = float(np.mean(val_losses))
        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: best_state was never set.")

    # -------- Test (original scale) --------
    model.load_state_dict(best_state)
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    yhat_s = np.vstack(preds)
    yhat = y_scaler.inverse_transform(yhat_s)
    ytrue = y_te

    rmse = np.sqrt(((yhat - ytrue) ** 2).mean(axis=0))
    print(f"test RMSE  left={rmse[0]:.4f}  right={rmse[1]:.4f}")

    # -------- Save best model + scalers --------
    torch.save(best_state, MODEL_PATH)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    print(f"saved: {MODEL_PATH}, {X_SCALER_PATH}, {Y_SCALER_PATH}")


if __name__ == "__main__":
    main()
