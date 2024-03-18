import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def laplacian2D(M,dx):
    return (
        - 2 * M + np.roll(M,shift=+1,axis=0) + np.roll(M,shift=-1,axis=0) # second derivative in x
        - 2 * M + np.roll(M,shift=+1,axis=1) + np.roll(M,shift=-1,axis=1) # second derivative in y
    ) / (dx ** 2)
    
def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear')
    ax.set_axis_off()


#Simulation Parameters
Lx = 120
Ly = 40
dx = 1 #.*(10**3) / size  # space step
dt = .01  # time step
T = dt*50000 #total duration of the simulation
n = int(T / dt)  # number of iterations

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
step_plot = n // 5

#Simulation Parameters
Da = 0.008
Ds = 0.16
ka = 0.0025
ks = 0.003
ha = 0.00025
hs = 0.003

A = np.random.rand(Lx, Ly)
S = np.random.rand(Lx, Ly)

#Simulation Loop 
dt = 1
dx = 1

for i in tqdm(range(n), total=n):
    
    # We update the variables.
    DA = laplacian2D(A,dx)
    DS = laplacian2D(S,dx)
    
    dAdt = Da * DA + ka * (S * (A**2) - A) + ha
    dSdt = Ds * DS - ks * (S * (A**2)) + hs
    
    A += dAdt * dt
    S += dSdt * dt
    

    # We plot the state of the system at different times.
    if i % step_plot == 0 and i < 5 * step_plot:
        ax = axes[0, i // step_plot]
        show_patterns(A, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        print(np.max(A)-np.min(A),np.max(S)-np.min(S))
        
        ax = axes[1, i // step_plot]
        show_patterns(S, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        
plt.show()