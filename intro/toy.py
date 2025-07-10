#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# one time-step
def dS(S, I, R):
    return -beta * ((S * I) / N)
def dI(S, I, R):
    return beta * ((S * I) / N) - I/D
def dR(S, I, R):
    return I/D

# made up parameters
beta = 0.6
D = 5

# starting conditions
N = 1000
S = 1000
I = 1
R = 0

# simulation loop
s = [S]
i = [I]
r = [R]
t = 50
for j in range(t):
    S = S + dS(s[-1], i[-1], r[-1])
    I = I + dI(s[-1], i[-1], r[-1])
    R = R + dR(s[-1], i[-1], r[-1])
    s.append(S)
    i.append(I)
    r.append(R)
    if(j % 5 == 0):
        print("S:", S, "I:", I, "R:", R)

fig, ax = plt.subplots()
ax.plot(range(len(s)), s, label="Susceptible")
ax.plot(range(len(i)), i, label="Infected")
ax.plot(range(len(r)), r, label="Recovered")
ax.legend()
plt.show()