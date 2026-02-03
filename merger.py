import torch
import torch.nn as nn
import joblib
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- networks (must match training exactly) ----------
class MultiplierNet(nn.Module):
    def __init__(self, in_dim=5, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

class WheelsNet(nn.Module):
    def __init__(self, in_dim=5, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

# ---------- helper: store StandardScaler inside torch module ----------
def scaler_to_tensors(scaler):
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    return mean, scale

class CombinedPolicy(nn.Module):
    """
    Input:  (N,5) raw features: [x,y,theta,x_ref,y_ref]
    Output: (N,2) final wheel cmds = wheel_cmds * multiplier
    """
    def __init__(
        self,
        wheels_state_path="policy_bc2.pt",
        mult_state_path="policy_bc j.pt",
        x_scaler_wheels_path="x_scaler2.pkl",
        y_scaler_wheels_path="y_scaler2.pkl",
        x_scaler_mult_path="x_scaler j.pkl",
        y_scaler_mult_path="y_scaler j.pkl",
    ):
        super().__init__()

        # load sklearn scalers
        x_sw = joblib.load(x_scaler_wheels_path)
        y_sw = joblib.load(y_scaler_wheels_path)
        x_sm = joblib.load(x_scaler_mult_path)
        y_sm = joblib.load(y_scaler_mult_path)

        # register as buffers so they move with .to(device) and save in one file
        xw_mean, xw_scale = scaler_to_tensors(x_sw)
        yw_mean, yw_scale = scaler_to_tensors(y_sw)
        xm_mean, xm_scale = scaler_to_tensors(x_sm)
        ym_mean, ym_scale = scaler_to_tensors(y_sm)

        self.register_buffer("xw_mean", xw_mean)
        self.register_buffer("xw_scale", xw_scale)
        self.register_buffer("yw_mean", yw_mean)
        self.register_buffer("yw_scale", yw_scale)

        self.register_buffer("xm_mean", xm_mean)
        self.register_buffer("xm_scale", xm_scale)
        self.register_buffer("ym_mean", ym_mean)
        self.register_buffer("ym_scale", ym_scale)

        # load nets + weights
        self.wheels = WheelsNet()
        self.mult = MultiplierNet()

        self.wheels.load_state_dict(torch.load(wheels_state_path, map_location="cpu"))
        self.mult.load_state_dict(torch.load(mult_state_path, map_location="cpu"))

        self.wheels.eval()
        self.mult.eval()

    @torch.no_grad()
    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (N,5) float32 raw input
        """
        # normalize separately (because you trained with separate x_scalers)
        x_w = (x_raw - self.xw_mean) / self.xw_scale
        x_m = (x_raw - self.xm_mean) / self.xm_scale

        # predict in scaled space
        wheels_scaled = self.wheels(x_w)        # (N,2)
        mult_scaled = self.mult(x_m)            # (N,1)

        # inverse-transform to real units
        wheels_real = wheels_scaled * self.yw_scale + self.yw_mean     # (N,2)
        mult_real = mult_scaled * self.ym_scale + self.ym_mean         # (N,1)

        # multiply (broadcast multiplier across 2 wheel outputs)
        out = wheels_real * mult_real
        return out


if __name__ == "__main__":
    model = CombinedPolicy().to(DEVICE)
    model.eval()

    # Example inference with one sample
    x = torch.tensor([[1.0, 2.0, 0.1, 5.0, 6.0]], dtype=torch.float32, device=DEVICE)
    y = model(x)
    print("final wheel cmds:", y)

    # Save ONE file that includes scalers + both nets
    torch.save(model.state_dict(), "combined_model_state.pt")
    scripted = torch.jit.script(model.cpu())
    scripted.save("combined_model_scripted.pt")
    print("saved: combined_model_state.pt and combined_model_scripted.pt")
