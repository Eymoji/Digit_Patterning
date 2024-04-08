import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from outils import *
import imageio
import os

digit, Lx, Ly = matrix_from_black_white_png('hand2.png')
borders = digit_borders(digit)[:,:,np.newaxis]

#Simulation Parameters
# Lx, Ly = 180, 24    # size of the 2D grid
dx = 1              # space step
dt = 20             # time step
T = 120000          # total duration of the simulation
n = int(T / dt) + 1 # number of iterations

#fig, axes = plt.subplots(4, 5, figsize=(20, 8))
fig, axes = plt.subplots(1, 2, figsize=(30, 10))
step_plot = n // 5
plot_name = 'HANDplot.png'
gif_images_folder = 'gif_images/'
# Create the folder if it does not exist
if not os.path.exists(gif_images_folder):
    os.makedirs(gif_images_folder)
gif_name = 'HANDgif.gif'
step_gif = n // 50

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

ratio_random = 0.01
L0 = 40         # initial length of the digit
Lfin = 150      # final length of the digit
T0 = T // 3     # time to start the digit growth
W = 20          # width of the digit (in pixels)
e = 10          # roundness of the digit tip
lp = 50         # limit distance of growth from the tip of the digit (in pixels)

# digit = draw_ellipse_mask(Lx, Ly, Lfin, W, e, 0)


A = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
S = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
B = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
I = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
    
for i in tqdm(range(n), total=n):
    t = i * dt
    L = L0 if t < T0 else L0 + (Lfin - L0) * (t - T0) / (T - T0)
    DL = L - L0
    # circle mask around the tip of the digit of coordinates (Lx//2 - DL//2, Ly//2) of radius lp
    mask = np.ones((Lx, Ly))# if t < T0 else draw_circle_mask(Lx, Ly, Lx//2 - DL//2 - L//2, Ly//2, lp)
    A = diffuse_2d(A, Da, dt) * mask + (1-mask) * A
    S = diffuse_2d(S, Ds, dt) * mask + (1-mask) * S
    B = diffuse_2d(B, Db, dt) * mask + (1-mask) * B
    I = diffuse_2d(I, Di, dt) * mask + (1-mask) * I
    dAdt = digit * (ka * (S * (A**2) - A) + ha) + (1 - digit) * (- kdeg * A)
    dSdt = digit * (- ks * (S * (A**2)) + hs)
    dBdt = digit * (kb * (S ** 2) * (B**2 / I + hb) / (1 + kappab * A * B ** 2) - kb * B)
    dIdt = digit * (ki * (B ** 2 - I))
    A += dAdt * dt * mask
    S += dSdt * dt * mask
    B += dBdt * dt * mask
    I += dIdt * dt * mask
    # We plot the state of the system at different times.
    if i % step_gif == 0:
        AB1 = np.ones((Lx, Ly, 3), dtype = float)
        AB2 = np.ones((Lx, Ly, 3), dtype = float)
        AB1[:,:,0] -= np.power(normalize(B), 1)
        AB1[:,:,1] -= np.power(normalize(B), 1)
        AB2[:,:,0] -= np.power(normalize(B), 1)
        AB2[:,:,1] -= np.power(normalize(B), 1)
        AB2[:,:,1] -= np.power(normalize(A), 1)
        AB2[:,:,2] -= np.power(normalize(A), 1)
        # contrast the image
        AB1 = contrast_rgb(AB1, 1.5)
        AB2 = contrast_rgb(AB2, 1.5)
        
        
        AB1 += (1-digit[:,:,np.newaxis])
        AB1 -= borders
        AB2 += (1-digit[:,:,np.newaxis])
        AB2 -= borders
        # save the image with rbg colors
        # save_rgb_image_plt(AB, gif_images_folder + f'{i}.png')
        if i % step_plot == 0:
            ax = axes[0]
            ax.imshow(AB1)
            ax.set_axis_off()
            ax = axes[1]
            ax.imshow(AB2)
            ax.set_axis_off()

# save the plot fig
plt.tight_layout()
fig.savefig(plot_name + '.png')
plt.close()