from autograd import Value
import pandas as pd
import matplotlib.pyplot as plt

# one time-step
def dS_core(S, I):
    return -beta*S*I/core_population
def dI_core(S, I):
    return beta*S*I/core_population # - gamma*I
def dI_peripheral(I_core, I_peripheral): # "the bridge"
    return alpha1*I_core + alpha2*I_peripheral # - gamma*I_peripheral

# facts
population = Value(7500000)
core_population = Value(90000)
span = Value(161)

# load data from csv
data = pd.read_csv('hist.csv')
F = data['F'].values[:span.data]
F = [Value(x) for x in F]
M = data['M'].values[:span.data]
M = [Value(x) for x in M]
M_cummulative = [Value(0)]
for i in range(len(M)):
    M_cummulative.append(M_cummulative[-1] + M[i])
M_cummulative = M_cummulative[1:]  
F_cummulative = [Value(0)]
for i in range(len(F)):
    F_cummulative.append(F_cummulative[-1] + F[i])
F_cummulative = F_cummulative[1:]

# core group: MSM
# 72% of male infections are MSM
core = [x * Value(0.72) for x in M_cummulative]
peripheral = [x * Value(0.28) + y for x, y in zip(M_cummulative, F_cummulative)]

# target parameters
alpha1 = Value(0)  # rate of infection from core to peripheral
alpha2 = Value(0)  # rate of infection from peripheral to peripheral
beta = Value(0.001)  # infection rate (core to core)
gamma = Value(0.1)  # dying rate

def forward(S_core, I_core, I_peripheral):
    s_core = []
    i_core = []
    i_peripheral = []
    for _ in range(span.data):
        ds_core = dS_core(S_core, I_core)
        di_core = dI_core(S_core, I_core)
        di_peripheral = dI_peripheral(I_core, I_peripheral)

        # update all simultaneously
        S_core = S_core + ds_core
        I_core = I_core + di_core
        I_peripheral = I_peripheral + di_peripheral

        s_core.append(S_core)
        i_core.append(I_core)
        i_peripheral.append(I_peripheral)
    return s_core, i_core, i_peripheral

# calculate loss
def loss(i_core, i_peripheral, core, peripheral):
    loss_value = Value(0)
    for i in range(len(F)):
        loss_value += (i_core[i] - core[i]) ** 2
        loss_value += (i_peripheral[i] - peripheral[i]) ** 2
        print(loss_value.data)
    return (loss_value/len(core))**(1/2) #RMSE

# draw desease spread
def plot(s, i_core, i_peripheral, core, peripheral):
    fig, ax = plt.subplots()
    ax.plot(range(len(s)), [v.data for v in s], label="Susceptible (core)")
    ax.plot(range(len(i_core)), [v.data for v in i_core], label="Estimated core")
    ax.plot(range(len(i_peripheral)), [v.data for v in i_peripheral], label="Estimated peripheral")
    ax.plot(range(len(core)), [v.data for v in core], label="Actual core")
    ax.plot(range(len(peripheral)), [v.data for v in peripheral], label="Actual peripheral")
    ax.legend()
    plt.show()

# optimization loop
lr = 0.001  # learning rate
for _ in range(2):
    I_core = core[0]
    I_peripheral = peripheral[0]
    S_core = core_population - I_core

    # forward pass
    s_core, i_core, i_peripheral = forward(S_core, I_core, I_peripheral)
    
    # calculate loss
    l = loss(i_core, i_peripheral, core, peripheral)

    # backward pass
    l.grad = 1
    l.backward()
    
    # update parameters
    beta.data -= lr * beta.grad
    alpha1.data -= lr * alpha1.grad
    alpha2.data -= lr * alpha2.grad
    gamma.data -= lr * gamma.grad

    # reset gradients for next iteration
    beta.grad = 0
    alpha1.grad = 0
    alpha2.grad = 0
    gamma.grad = 0

    # print progress
    if _ % 10 == 0:
        print(f"Iteration {_}: loss = {l.data}")

# print results
print("Optimized alpha1:", alpha1.data)
print("Optimized alpha2:", alpha2.data)
print("Optimized beta:", beta.data)
print("Optimized gamma:", gamma.data)

# compare with real data
fig, ax = plt.subplots()
ax.plot(range(len(i_core)), [v.data for v in i_core], label="Estimation (Core)")
ax.plot(range(len(core)), [v.data for v in core], label="Data (Core)")
ax.plot(range(len(i_peripheral)), [v.data for v in i_peripheral], label="Estimation (Peripheral)")
ax.plot(range(len(peripheral)), [v.data for v in peripheral], label="Data (Peripheral)")
ax.legend()
plt.show()