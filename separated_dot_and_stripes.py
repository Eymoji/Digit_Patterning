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

fig, axes = plt.subplots(2, 2, figsize=(20, 8))

#Simulation Parameters
Da = 0.016
Ds = 0.32
Db = 6.e-3
Di = 0.18
ka = 0.0025
ks = 0.003
kb = 0.01875
kdeg = 1.e-5
kappab = 0.2
ki = 0.0375
ha = 0.00025
hb = 0.00187
hs = 0.003

#A = np.ones((Lx, Ly))*0.1
#S = np.ones((Lx, Ly))*0.1
#B = np.ones((Lx, Ly))*0.1
#I = np.ones((Lx, Ly))*0.1

A = (np.random.rand(Lx, Ly)-0.5)*0.001+1
S = (np.random.rand(Lx, Ly)-0.5)*0.001+1
B = (np.random.rand(Lx, Ly)-0.5)*0.001+1
I = (np.random.rand(Lx, Ly)-0.5)*0.001+1

A0 = A.copy()
S0 = S.copy()
B0 = B.copy()
I0 = I.copy()

#for i in tqdm(range(n), total=n):
for i in tqdm(range(n), total=n):
    
      A = diffuse_2d(A, Da, dt)
      S = diffuse_2d(S, Ds, dt)
      B = diffuse_2d(B, Db, dt)
      I = diffuse_2d(I, Di, dt)
    
      dAdt = ka * (S * A**2 - A) + ha
      dSdt = - ks * S * A**2 + hs
      dBdt = kb * S0**2 * ((B**2 / I) + hb) / (1 + kappab * A0 * B**2) - kb * B
      dIdt = ki * (B**2 - I)
    
      A += dAdt * dt
      S += dSdt * dt
      B += dBdt * dt
      I += dIdt * dt

ax = axes[0, 0]
ax.imshow(A0, cmap='Oranges',
      interpolation='bilinear', vmin = np.min(A), vmax = np.max(A))
ax.set_axis_off()
        
ax = axes[1, 0]
ax.imshow(B0, cmap='Greens',
      interpolation='bilinear', vmin = np.min(B), vmax = np.max(B))
ax.set_axis_off()

ax = axes[0, 1]
ax.imshow(A, cmap='Oranges',
      interpolation='bilinear', vmin = np.min(A), vmax = np.max(A))
ax.set_axis_off()

ax = axes[1, 1]
ax.imshow(B, cmap='Greens',
      interpolation='bilinear', vmin = np.min(B), vmax = np.max(B))
ax.set_axis_off()

plt.savefig("Da_0016_Ds_032_separated_dot_and_stripes.png")
plt.show()
