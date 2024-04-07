import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from outils import *
import imageio
import os


#Simulation Parameters
Lx, Ly = 220, 24    # size of the 2D grid
dx = 1              # space step
dt = 20             # time step
T = 60000           # total duration of the simulation
n = int(T / dt) + 1 # number of iterations
ratio_random = 0.01 # ratio of random noise in the initial conditions

#fig, axes = plt.subplots(4, 5, figsize=(20, 8))
fig, axes = plt.subplots(1, 4, figsize=(10, 15))
step_plot = n // 5

plot_name = '2C.png'
gif_name = '2C.gif'
gif_images_folder = 'gif_images/'

# Create the folder if it does not exist
if not os.path.exists(gif_images_folder):
    os.makedirs(gif_images_folder)
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


L0 = 10         # initial length of the digit
Lfin = 100      # final length of the digit
T0 = 5e3        # time to start the digit growth
W = 20          # width of the digit (in pixels)
e = 10          # roundness of the digit tip
lp = 50         # limit distance of growth from the tip of the digit (in pixels)


for ip, lp in enumerate([100]):
    A = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
    S = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
    B = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
    I = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2

    for i in tqdm(range(n), total=n):
        t = i * dt
        L = L0 if t < T0 else L0 + (Lfin - L0) * (t - T0) / (T - T0)
        DL = L - L0

        digit = draw_ellipse_mask(Lx, Ly, L, W, e, DL)

        # circle mask around the tip of the digit of coordinates (Lx//2 - DL//2, Ly//2) of radius lp
        mask = np.ones((Lx, Ly)) if t < T0 else draw_circle_mask(Lx, Ly, Lx//2 - DL//2 - L//2, Ly//2, lp)

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
        
        if i % step_gif == 0:
            AB = np.ones((Lx, Ly, 3), dtype = float)
            AB[:,:,0] -= np.power(normalize(B), 1)
            AB[:,:,1] -= np.power(normalize(B), 1)
            AB = contrast_rgb(AB, 1.5)
            AB += (1-digit[:,:,np.newaxis])
            # save the image with rbg colors
            save_rgb_image_plt(AB, gif_images_folder + f'{i}.png')  
    
    ax = axes[ip]
    ax.imshow(AB)
    ax.set_axis_off()
    
    # Create the gif
    images = []
    for i in range(0, n, step_gif):
        images.append(imageio.imread(gif_images_folder + f'{i}.png'))
    imageio.mimsave(gif_name + f'_lp{lp}.gif', images)

    # Remove the images of the gif
    for i in range(0, n, step_gif):
       os.remove(gif_images_folder + f'{i}.png')
# save the plot fig
plt.tight_layout()
plt.savefig(plot_name)
plt.close()