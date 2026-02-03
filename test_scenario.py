import math
import cv2
import numpy as np
import torch

from world import world


# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

GOAL_TOL = 5.0


# ===================== LOAD MODEL (TorchScript) =====================
# This is the output of the previous code: scripted.save("combined_policy_scripted.pt")
model = torch.jit.load("combined_policy_scripted.pt", map_location=DEVICE)
model.eval()


# ===================== INFERENCE =====================
def combined_cmd(x, y, theta, x_ref, y_ref):
    x_in = np.array([[x, y, theta, x_ref, y_ref]], dtype=np.float32)
    x_t = torch.from_numpy(x_in).to(DEVICE)

    with torch.no_grad():
        out = model(x_t).detach().cpu().numpy()[0]  # (2,)

    return float(out[0]), float(out[1])


# ===================== SIM =====================
robot = world(30)

targets = [
    (-50, 50),
    (-50, 100),
    (-50, 150),
    (-50, 200),
    (-50, 250)
]

for target in targets:
    while True:
        frame = robot.visualize()
        frame = robot.draw_target([target], frame)
        cv2.imshow("robot", frame)
        cv2.waitKey(1)

        rx, ry,_ = robot.pos()
        dist = math.hypot(target[0] - rx, target[1] - ry)

        if dist < GOAL_TOL:
            robot.tick(0.0, 0.0)
            break

        left_cmd, right_cmd = combined_cmd(robot.x, robot.y, robot.theta, target[0], target[1])
        robot.tick(left_cmd, right_cmd)
