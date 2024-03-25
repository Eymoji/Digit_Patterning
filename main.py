import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from outils import *


#Simulation Parameters
Lx, Ly = 128, 18    # size of the 2D grid
dx = 1              # space step
dt = 20              # time step
T = 60000        # total duration of the simulation
n = int(T / dt)     # number of iterations

#fig, axes = plt.subplots(4, 5, figsize=(20, 8))
fig, axes = plt.subplots(1, 5, figsize=(10, 15))
step_plot = n // 5

# Reaction Parameters
Da = 0.008
Ds = 0.16
Db = 6.e-3
Di = 0.12
ka = 0.0025
ks = 0.003
kb = 0.01875
ki = 0.0375
kdeg = 1.e-5
kappab = 0.2
ha = 0.00025
hb = 0.00187
hs = 0.003

A = np.random.rand(Lx, Ly)
S = np.random.rand(Lx, Ly)
B = np.random.rand(Lx, Ly)
I = np.random.rand(Lx, Ly)
A = np.ones((Lx, Ly))
S = np.ones((Lx, Ly))
B = np.ones((Lx, Ly))
I = np.ones((Lx, Ly))
digit = draw_ellipse_mask(Lx, Ly, Lx//2-2-Ly//2-2, Ly//2-2)
# digit = np.ones((Lx, Ly))

for i in tqdm(range(n), total=n):
    
    A = diffuse_2d(A, Da, dt)
    S = diffuse_2d(S, Ds, dt)
    B = diffuse_2d(B, Db, dt)
    I = diffuse_2d(I, Di, dt)
    
    dAdt = digit * (ka * (S * (A**2) - A) + ha) + (1 - digit) * (- kdeg * A)
    dSdt = digit * (- ks * (S * (A**2)) + hs)
    dBdt = digit * (kb * (S ** 2) * (B**2 / I + hb) / (1 + kappab * A * B ** 2) - kb * B)
    dIdt = digit * (ki * (B ** 2 - I))
    
    A += dAdt * dt
    S += dSdt * dt
    B += dBdt * dt
    I += dIdt * dt
    
    
    # We plot the state of the system at different times.
    if i % step_plot == 0 and i < 5 * step_plot:
        
        AB = np.ones((Lx, Ly, 3), dtype = float)
        AB[:,:,0] -= normalize(B)
        AB[:,:,1] -= normalize(B)
        AB[:,:,1] -= normalize(A)
        AB[:,:,2] -= normalize(A)
        AB *= digit[:,:,np.newaxis] + AB / 2
        
        ax = axes[i // step_plot]
        ax.imshow(AB)
        ax.set_axis_off()
        
        
        
plt.show()
