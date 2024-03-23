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

def draw_ellipse_mask(Lx, Ly, a, b):
    E = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            if (np.abs(Lx//2 - i) <= a and np.abs(Ly//2 - j) <= b) or (
                np.sqrt((a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b) or (
                np.sqrt((-a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b):
                E[i,j] = 1
    return E


#Simulation Parameters
Lx = 128
Ly = 24
dx = 1 #.*(10**3) / size  # space step
dt = .01  # time step
T = dt*50000 #total duration of the simulation
n = int(T / dt)  # number of iterations

fig, axes = plt.subplots(4, 5, figsize=(20, 8))
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

A = np.random.rand(Lx, Ly)
S = np.random.rand(Lx, Ly)
B = np.random.rand(Lx, Ly)
I = np.random.rand(Lx, Ly)

#Simulation Loop 
dt = 1
dx = 1

digit = draw_ellipse_mask(Lx, Ly, Lx//3, Ly//3)

for i in tqdm(range(n), total=n):
    
    # We update the variables.
    DA = laplacian2D(A,dx)
    DS = laplacian2D(S,dx)
    DB = laplacian2D(B,dx)
    DI = laplacian2D(I,dx)
    
    dAdt = Da * DA + digit * (ka * (S * (A**2) - A) + ha) + (1 - digit) * (- kdeg * A)
    dSdt = Ds * DS + digit * (- ks * (S * (A**2)) + hs)
    dBdt = Db * DB + digit * (kb * (S ** 2) * (B**2 / I + hb) / (1 + kappab * A * B ** 2) - kb * B)
    dIdt = Di * DI + digit * (ki * (B ** 2 - I))
    
    A += dAdt * dt
    S += dSdt * dt
    B += dBdt * dt
    I += dIdt * dt
    

    # We plot the state of the system at different times.
    if i % step_plot == 0 and i < 5 * step_plot:
        ax = axes[0, i // step_plot]
        show_patterns(A*digit, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        
        ax = axes[1, i // step_plot]
        show_patterns(S*digit, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        
        ax = axes[2, i // step_plot]
        show_patterns(B*digit, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        
        ax = axes[3, i // step_plot]
        show_patterns(I*digit, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        
plt.show()