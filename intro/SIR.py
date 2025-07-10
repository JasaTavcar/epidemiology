#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from autograd import Value

# one time-step
def dS(S, I, R):
    return -beta*S*I/N
def dI(S, I, R):
    return beta*S*I/N - I/D
def dR(S, I, R):
    return I/D

# facts
N = Value(763)
days = Value(25) 
in_bed = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(in_bed)):
    in_bed[i] = Value(in_bed[i])

# target parameters
beta = Value(1)  # infection rate
D = Value(1)
R_init = Value(150)

def forward(S, I, R):
    s = []
    i = []
    r = []
    for _ in range(days.data):
        dS_val = dS(S, I, R)
        dI_val = dI(S, I, R)
        dR_val = dR(S, I, R)

        # update all simultaneously
        S = S + dS_val
        I = I + dI_val
        R = R + dR_val

        s.append(S)
        i.append(I)
        r.append(R)
    return s, i, r

# calculate loss
def loss(s, i, r):
    loss_value = Value(0)
    for j in range(len(in_bed)):
        if j < len(i):
            loss_value += (i[j] - in_bed[j]) ** 2
    return (loss_value/len(in_bed))**(1/2) #RMSE

# draw desease spread
def plot(s, i, r):
    fig, ax = plt.subplots()
    ax.plot(range(len(s)), [v.data for v in s], label="Susceptible")
    ax.plot(range(len(i)), [v.data for v in i], label="Infected")
    ax.plot(range(len(r)), [v.data for v in r], label="Recovered")
    ax.legend()
    plt.show()

# optimization loop
lr = 0.001  # learning rate
for _ in range(200):
    R = R_init
    I = in_bed[0]
    S = N - I - R

    # forward pass
    s, i, r = forward(S, I, R)
    
    # calculate loss
    l = loss(s, i, r)
    
    # backward pass
    l.grad = 1
    l.backward()
    
    # update parameters
    beta.data -= lr * beta.grad
    D.data -= lr * D.grad
    R_init.data -= lr * R_init.grad

    # reset gradients for next iteration
    S.grad = 0
    I.grad = 0
    R.grad = 0
    beta.grad = 0
    D.grad = 0

    # print progress
    if _ % 10 == 0:
        print(f"Iteration {_}: loss = {l.data}")

# print results
print("Optimized beta:", beta.data)
print("Optimized D:", D.data)
print("Optimized R_init:", R_init.data)
print("Final loss:", l.data)
plot(s, i, r)

# compare with real data
fig, ax = plt.subplots()
ax.plot(range(len(i)), [v.data for v in i], label="Estimation")
ax.plot(range(len(in_bed)), [v.data for v in in_bed], label="Data")
ax.legend()
plt.show()