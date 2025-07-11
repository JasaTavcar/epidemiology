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

# Initial conditions
core_population = 90000.0

# Time points
t = torch.linspace(0, span - 1, span)

# --- ODE model ---
class HIVModel(torch.nn.Module):
    def __init__(self, beta=0.1, alpha1=0.1, alpha2=0.1, alpha3=0.1, gamma=0.1, core_fraction_init=0.72):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.alpha1 = torch.nn.Parameter(torch.tensor(alpha1))  # core to women
        self.alpha2 = torch.nn.Parameter(torch.tensor(alpha2))  # women to men
        self.alpha3 = torch.nn.Parameter(torch.tensor(alpha3))  # men to women
        self.gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.core_fraction_logit = torch.nn.Parameter(torch.logit(torch.tensor(core_fraction_init)))

    @property
    def core_fraction(self):
        return torch.sigmoid(self.core_fraction_logit)

    @property
    def peripheral_fraction(self):
        return 1.0 - self.core_fraction

    def forward(self, t, y):
        S, I, C, Ipm, Cpm, Ipw, Cpw = y
        # Core infection
        infection_core = self.beta * S * I / core_population
        # Core to women
        infection_core_to_women = self.alpha1 * I
        # Women to men
        infection_women_to_men = self.alpha2 * Ipw
        # Men to women
        infection_men_to_women = self.alpha3 * Ipm

        dS = -infection_core
        dI = infection_core - self.gamma * I
        dC = infection_core
        dIpm = infection_women_to_men - self.gamma * Ipm
        dCpm = infection_women_to_men
        dIpw = infection_core_to_women + infection_men_to_women - self.gamma * Ipw
        dCpw = infection_core_to_women + infection_men_to_women

        return torch.stack([dS, dI, dC, dIpm, dCpm, dIpw, dCpw])


def train(beta, alpha1, alpha2, alpha3, gamma, core_fraction_init=0.72, verbose=False):
    model = HIVModel(beta, alpha1, alpha2, alpha3, gamma, core_fraction_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1500):
        optimizer.zero_grad()
        core_frac = model.core_fraction
        peripheral_frac = model.peripheral_fraction

        # Data splits
        core = core_frac * M_cum
        peripheral_men = peripheral_frac * M_cum
        peripheral_women = F_cum
        core_new = M * core_frac
        peripheral_men_new = M * peripheral_frac
        peripheral_women_new = F

        # Initial conditions
        I0 = core[0]
        Ipm0 = peripheral_men[0]
        Ipw0 = peripheral_women[0]
        S0 = core_population - I0
        C_core0 = core[0]
        C_pm0 = peripheral_men[0]
        C_pw0 = peripheral_women[0]
        y0 = torch.tensor([S0, I0, C_core0, Ipm0, C_pm0, Ipw0, C_pw0])

        # Integrate ODE
        sol = odeint(model, y0, t, method='dopri5')
        S_pred, I_pred, C_pred, Ipm_pred, Cpm_pred, Ipw_pred, Cpw_pred = sol.T

        # Compute daily new infections
        daily_core_new = torch.zeros_like(C_pred)
        daily_core_new[1:] = C_pred[1:] - C_pred[:-1]
        daily_pm_new = torch.zeros_like(Cpm_pred)
        daily_pm_new[1:] = Cpm_pred[1:] - Cpm_pred[:-1]
        daily_pw_new = torch.zeros_like(Cpw_pred)
        daily_pw_new[1:] = Cpw_pred[1:] - Cpw_pred[:-1]

        # RMSE loss
        loss = torch.sqrt(
            torch.mean((daily_core_new - core_new) ** 2 +
                       (daily_pm_new - peripheral_men_new) ** 2 +
                       (daily_pw_new - peripheral_women_new) ** 2)
        )
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.3f}, Core Fraction = {core_frac.item():.3f}")
    return model, (S_pred, I_pred, C_pred, Ipm_pred, Cpm_pred, Ipw_pred, Cpw_pred), loss

# Main block
if __name__ == "__main__":
    beta = 0.2
    alpha1 = 0.1
    alpha2 = 0.1
    alpha3 = 0.1
    gamma = 0.8
    core_fraction = 0.5
    model, results, loss = train(beta, alpha1, alpha2, alpha3, gamma, core_fraction, verbose=True)
    S_pred, I_pred, C_pred, Ipm_pred, Cpm_pred, Ipw_pred, Cpw_pred = results
    final_core_frac = model.core_fraction.item()
    final_peripheral_frac = model.peripheral_fraction.item()
    core = final_core_frac * M_cum
    peripheral_men = final_peripheral_frac * M_cum
    peripheral_women = F_cum

    # --- Cumulative Plot ---
    plt.figure(figsize=(12,8))
    plt.subplot(2, 1, 1)
    plt.plot(core.numpy(), label=f"Core (reported, frac={final_core_frac:.3f})")
    plt.plot(peripheral_men.numpy(), label=f"Peripheral men (reported, frac={final_peripheral_frac:.3f})")
    plt.plot(peripheral_women.numpy(), label="Peripheral women (reported)")
    plt.plot(C_pred.detach().numpy(), label="Core (estimated)")
    plt.plot(Cpm_pred.detach().numpy(), label="Peripheral men (estimated)")
    plt.plot(Cpw_pred.detach().numpy(), label="Peripheral women (estimated)")
    plt.legend()
    plt.title("HIV Spread Model - Results (Cumulative Infections)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Infections")
    plt.grid(True)

    # --- Infection Plot ---
    plt.subplot(2, 1, 2)
    daily_core = np.zeros_like(C_pred.detach().numpy())
    daily_core[1:] = C_pred.detach().numpy()[1:] - C_pred.detach().numpy()[:-1]
    daily_pm = np.zeros_like(Cpm_pred.detach().numpy())
    daily_pm[1:] = Cpm_pred.detach().numpy()[1:] - Cpm_pred.detach().numpy()[:-1]
    daily_pw = np.zeros_like(Cpw_pred.detach().numpy())
    daily_pw[1:] = Cpw_pred.detach().numpy()[1:] - Cpw_pred.detach().numpy()[:-1]
    plt.plot(daily_core, label="Daily New Infections Core (Model)")
    plt.plot(daily_pm, label="Daily New Infections Peripheral Men (Model)")
    plt.plot(daily_pw, label="Daily New Infections Peripheral Women (Model)")
    plt.plot(M.numpy() * final_core_frac, label="Daily New Infections Core")
    plt.plot(M.numpy() * final_peripheral_frac, label="Daily New Infections Peripheral Men")
    plt.plot(F.numpy(), label="Daily New Infections Peripheral Women")
    plt.legend()
    plt.title("Quarterly New Infections")
    plt.xlabel("Time")
    plt.ylabel("New Infections")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL MODEL SUMMARY")
    print("=" * 80)
    print(f"Final loss: {loss.item():.6f}")
    print(f"Final parameters:")
    print(f"  beta: {model.beta.item():.4f}")
    print(f"  alpha1 (core to women): {model.alpha1.item():.4f}")
    print(f"  alpha2 (women to men): {model.alpha2.item():.4f}")
    print(f"  alpha3 (men to women): {model.alpha3.item():.4f}")
    print(f"  gamma: {model.gamma.item():.4f}")
    print(f"  core_fraction: {final_core_frac:.4f}")
    print(f"  peripheral_fraction: {final_peripheral_frac:.4f}")
    print("=" * 80)