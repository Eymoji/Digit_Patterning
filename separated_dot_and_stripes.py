import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from outils import *


#Simulation Parameters
Lx = 128
Ly = 128
dx = 1 #.*(10**3) / size  # space step
dt = 20  # time step
T = 12e4 #total duration of the simulation
n = int(T / dt)  # number of iterations

#fig, axes = plt.subplots(4, 5, figsize=(20, 8))
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
step_plot = n // 5

#Simulation Parameters
Da = 0.008
Ds = 0.16
Db = 6.e-3
Di = 0.12
ka = 0.0025
ks = 0.003
kb = 0.01875
kdeg = 1.e-5
kappab = 0.2
ki = 0.0375
ha = 0.00025
hb = 0.00187
hs = 0.003

A = np.ones((Lx, Ly))*0.1
S = np.ones((Lx, Ly))*0.1
B = np.ones((Lx, Ly))*0.1
I = np.ones((Lx, Ly))*0.1

A = (np.random.rand(Lx, Ly)-0.5)*0.001+1
S = (np.random.rand(Lx, Ly)-0.5)*0.001+1
B = (np.random.rand(Lx, Ly)-0.5)*0.001+1
I = (np.random.rand(Lx, Ly)-0.5)*0.001+1

A0 = A.copy()
S0 = S.copy()

#for i in tqdm(range(n), total=n):
for i in tqdm(range(n), total=n):
    
    A = diffuse_2d(A, Da, dt)
    S = diffuse_2d(S, Ds, dt)
    B = diffuse_2d(B, Db, dt)
    I = diffuse_2d(I, Di, dt)
    
    dAdt = ka * (S * (A**2) - A) + ha
    dSdt = - ks * (S * (A**2)) + hs
    dBdt = kb * S0**2 * (B**2 / I + hb) / (1 + kappab * A0 * B**2) - kb * B
    dIdt = ki * (B ** 2 - I)
    
    A += dAdt * dt
    S += dSdt * dt
    B += dBdt * dt
    I += dIdt * dt

    # We plot the state of the system at different times.
    if i % step_plot == 0 and i < 5 * step_plot:
        ax = axes[0, i // step_plot]
        ax.imshow(A, cmap='Blues',
              interpolation='bilinear', alpha = 0.5)
        ax.set_axis_off()
        
        ax = axes[1, i // step_plot]
        ax.imshow(B, cmap='Reds',
              interpolation='bilinear', alpha = 0.5)
        ax.set_axis_off()
        
        
plt.show()
