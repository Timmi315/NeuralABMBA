import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt


class Physical(nn.Module):
    def __init__(self, external_data, dt):
        super().__init__()
        self.external_data = external_data
        self.dt = dt

    def set_params(self, params):
        self.params = params

    

    def forward(self, t, state):
        idx = int(t.item())
        print(f"forward called at t={t}, state={state}")  # <- SHOULD PRINT
        data = self.external_data[idx]
        out = self.step(state, data, self.params, self.dt)
        return torch.tensor(out, dtype=torch.float32).reshape(1)

class RC(Physical):
    def step(self, state, data, params, dt):
        # RC model step
        C, R = params
        T_amb, heat, solar = data
        print(dt/C * (T_amb-state[0]/R + heat + solar))
        dT = dt / C * ((T_amb - state[0]) / R + heat + solar)
        print(state[0] + dT)
        return [dT]


# Dummy external data [T_out, heat, solar]
external_data = torch.tensor([[20.0, 0.0, 0.0]] * 100, dtype=torch.float32)  # shape: [100, 3]

# Initial condition
y0 = torch.tensor([21.0], dtype=torch.float32)  # shape: [1]

# Time vector
t = torch.arange(0, 100, dtype=torch.float32)

# Parameters
C = torch.tensor(1.0)
R = torch.tensor(100)

# Build model
model = RC(external_data=external_data, dt=1.0)
model.set_params([C, R])


print(f"time.shape: {t.shape}")
print(f"y0.shape: {y0.shape}")
print(f"external_data.shape: {model.external_data.shape}")
# Integrate
trajectory = odeint(model, y0, t, method="euler", options={"step_size": 1.0})
print(trajectory)
print("trajectory shape:", trajectory.shape)  # Expect: [100, 1]
print(external_data.shape)
plt.plot(trajectory)
plt.show()