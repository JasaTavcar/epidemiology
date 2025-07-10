import torch
from torchdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load data ---
span = 161
data = pd.read_csv('hist.csv')
F = torch.tensor(data['F'].values[:span], dtype=torch.float32)
M = torch.tensor(data['M'].values[:span], dtype=torch.float32)

# Cumulative sums
F_cum = torch.cumsum(F, dim=0)
M_cum = torch.cumsum(M, dim=0)

# Split into core (MSM) and peripheral
core = 0.72 * M_cum
peripheral = 0.28 * M_cum + F_cum
core_new = M * 0.72
peripheral_new = F + M * 0.28

# Initial conditions
core_population = 90000.0
I0 = core[0]
Ip0 = peripheral[0]
S0 = core_population - I0
C_core0 = core[0]
C_peripheral0 = peripheral[0]

y0 = torch.tensor([S0, I0, C_core0, Ip0, C_peripheral0])

# Time points
t = torch.linspace(0, span - 1, span)

# --- ODE model ---
class HIVModel(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.alpha1 = torch.nn.Parameter(torch.tensor(alpha))
        self.alpha2 = torch.nn.Parameter(torch.tensor(alpha))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma))  # decay (death/recovery)

    def forward(self, t, y):
        S, I, C, Ip, Cp = y
        infection_core = self.beta * S * I / core_population
        infection_peripheral = self.alpha1 * I + self.alpha2 * Ip

        dS = -infection_core
        dI = infection_core - self.gamma * I
        dC = infection_core  
        dIp = infection_peripheral - self.gamma * Ip
        dCp = infection_peripheral  

        return torch.stack([dS, dI, dC, dIp, dCp])

def train(alpha, beta, gamma):
    model = HIVModel(beta, alpha, gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        optimizer.zero_grad()

        # Integrate the ODE
        sol = odeint(model, y0, t, method='dopri5')
        S_pred, I_pred, C_pred, Ip_pred, Cp_pred = sol.T

        # Compute daily new infections from cumulative predictions
        daily_core_new = torch.zeros_like(C_pred)
        daily_core_new[1:] = C_pred[1:] - C_pred[:-1]
        
        daily_peripheral_new = torch.zeros_like(Cp_pred)
        daily_peripheral_new[1:] = Cp_pred[1:] - Cp_pred[:-1]
        
        # Total daily new infections from model
        daily_total_new = daily_core_new + daily_peripheral_new
        
        # Actual daily new infections from data
        actual_daily_new = core_new + peripheral_new
        
        # Compute RMSE loss between predicted daily new infections and data
        loss = torch.sqrt(torch.mean((daily_core_new - core_new) ** 2 + (daily_peripheral_new - peripheral_new) ** 2))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.3f}")

    return model, (S_pred, I_pred, C_pred, Ip_pred, Cp_pred), loss

model, res, loss = train(0.1, 0.1, 1.0)
S_pred, I_pred, C_pred, Ip_pred, Cp_pred = res

# --- Cumulative Plot ---
plt.figure(figsize=(10,6))
plt.plot(core.numpy(), label="Core (reported)")
plt.plot(peripheral.numpy(), label="Peripheral (reported)")
plt.plot(C_pred.detach().numpy(), label="Core (estimated)")
plt.plot(Cp_pred.detach().numpy(), label="Peripheral (estimated)")
plt.legend()
plt.title("HIV Spread Model with Cumulative Tracking")
plt.xlabel("Time")
plt.ylabel("Cumulative Infections")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Infection Plot ---
plt.figure(figsize=(10,6))
daily = np.zeros_like(C_pred.detach().numpy())
daily[1:] = C_pred.detach().numpy()[1:] - C_pred.detach().numpy()[:-1]
daily_peripheral = np.zeros_like(Cp_pred.detach().numpy())
daily_peripheral[1:] = Cp_pred.detach().numpy()[1:] - Cp_pred.detach().numpy()[:-1]
plt.plot(daily_peripheral + daily, label="Daily New Infections (Model)")
plt.plot(F.numpy() + M.numpy(), label="Daily New Infections (Reported)")
plt.legend()
plt.title("Daily New Infections")
plt.xlabel("Time")
plt.ylabel("New Infections")
plt.grid(True)
plt.tight_layout()
plt.show()