import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from outils import *
import imageio
import os


#Simulation Parameters
Lx, Ly = 150, 24    # size of the 2D grid
dx = 1              # space step
dt = 20             # time step
T = 60000*2         # total duration of the simulation
n = int(T / dt) + 1 # number of iterations

#fig, axes = plt.subplots(4, 5, figsize=(20, 8))
fig, axes = plt.subplots(1, 6, figsize=(10, 15))
step_plot = n // 5
gif_images_folder = 'gif_images/'
# Create the folder if it does not exist
if not os.path.exists(gif_images_folder):
    os.makedirs(gif_images_folder)
gif_name = '1Ftip.gif'
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
A = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
S = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
B = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2
I = np.ones((Lx, Ly)) + ratio_random * np.random.rand(Lx, Ly) - ratio_random/2

digit = draw_ellipse_mask(Lx, Ly, L=120, W=20, e=10)
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
    if i % step_gif == 0:
        
        AB = np.ones((Lx, Ly, 3), dtype = float)
        AB[:,:,0] -= np.power(normalize(B), 1)
        AB[:,:,1] -= np.power(normalize(B), 1)
        AB[:,:,1] -= np.power(normalize(A), 1)
        AB[:,:,2] -= np.power(normalize(A), 1)
        # 
        # contrast the image
        AB = contrast_rgb(AB, 1.5)
        
        AB += (1-digit[:,:,np.newaxis])
        
        # save the image with rbg colors
        save_rgb_image_plt(AB, gif_images_folder + f'{i}.png')
        
        
        if i % step_plot == 0:
            ax = axes[i // step_plot]
            ax.imshow(AB)
            ax.set_axis_off()

# save the plot fig
plt.tight_layout()
plt.savefig('1Ftip.png')
plt.close()


images = []
for i in range(0, n, step_gif):
    images.append(imageio.imread(gif_images_folder + f'{i}.png'))
imageio.mimsave(gif_name, images)

# Remove the images of the gif
for i in range(0, n, step_gif):
    os.remove(gif_images_folder + f'{i}.png')