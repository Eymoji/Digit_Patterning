import numpy as np
from scipy.fft import dctn, idctn
from functools import lru_cache
import matplotlib.pyplot as plt

# def draw_ellipse_mask(Lx, Ly, L, W, e):
#     E = np.zeros((Lx, Ly))
#     for i in range(Lx):
#         for j in range(Ly):
#             IN_RECTANGLE = (np.abs(Lx//2 - i) < L//2) and (np.abs(Ly//2 - j) < W//2)
#             IN_PLUS_ELLIPSE = False
#             IN_MINUS_ELLIPSE = False
#             if e != 0 and W != 0:
#                 IN_PLUS_ELLIPSE = (np.sqrt((L//2 + Lx//2 - i)**2/e**2 + (Ly//2 - j)**2/(W//2)**2) < 1)
#                 IN_MINUS_ELLIPSE = (np.sqrt((-L//2 + Lx//2 - i)**2/e**2 + (Ly//2 - j)**2/(W//2)**2) < 1)
#             if IN_RECTANGLE or IN_PLUS_ELLIPSE or IN_MINUS_ELLIPSE:
#                 E[i,j] = 1
#     return E

@lru_cache(maxsize=128)
def draw_ellipse_mask(Lx, Ly, L, W, e, DL=0):
    Cx, Cy = Lx//2 - DL//2, Ly//2
    E = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            IN_RECTANGLE = (np.abs(Cx - i) < L//2) and (np.abs(Cy - j) < W//2)
            IN_PLUS_ELLIPSE = False
            IN_MINUS_ELLIPSE = False
            if e != 0 and W != 0:
                IN_PLUS_ELLIPSE = (np.sqrt((L//2 + Cx - i)**2/e**2 + (Cy - j)**2/(W//2)**2) < 1)
                IN_MINUS_ELLIPSE = (np.sqrt((-L//2 + Cx - i)**2/e**2 + (Cy - j)**2/(W//2)**2) < 1)
            if IN_RECTANGLE or IN_PLUS_ELLIPSE or IN_MINUS_ELLIPSE:
                E[i,j] = 1
    return E

@lru_cache(maxsize=128)
def draw_circle_mask(Lx, Ly, Cx, Cy, r):
    x, y = np.ogrid[:Lx, :Ly]
    return (x - Cx)**2 + (y - Cy)**2 < r**2




def normalize(M):
    return (M - np.min(M)) / (np.max(M) - np.min(M))

def show_patterns(U, color, ax=None):
    ax.imshow(U, cmap=color,
              interpolation='bilinear')
    ax.set_axis_off()

@lru_cache(maxsize=128)
def make_2d_counting_grid(Lx, Ly):
    M = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            M[i,j] = (i**2/Lx**2 + j**2/Ly**2)
    return M
    

def diffuse_2d(grid, diffusivity, dt):
    """
    Simulates 2D diffusion on a grid using backward Euler and DCT with reflective boundaries.
        
    Args:
      grid: A 2D numpy array representing the concentration field.
      diffusivity: The diffusion coefficient.
      dt: The time step.

    Returns:
        A 2D numpy array representing the concentration field after one time step.
    """
    Nx, Ny = grid.shape

    # Apply 2D DCT with proper normalization for Neumann (reflective) boundaries
    concentration_dct = dctn(grid, norm='ortho')
        
    # Update concentration in DCT domain (backward Euler)
    Lx, Ly = concentration_dct.shape
    # for i in range(Lx):
    #     for j in range(Ly):
    #         concentration_dct[i,j] = concentration_dct[i,j] / (1 + 2 * np.pi * diffusivity * dt * ((i/Lx)**2 + (j/Ly)**2))

    M = make_2d_counting_grid(Lx, Ly)
    concentration_dct = concentration_dct / (1 + 2 * np.pi * diffusivity * dt * M)
    # Inverse DCT to get updated concentration field
    updated_grid = idctn(concentration_dct, norm='ortho')

    return updated_grid


def duplicate_line(M, n):
    """
    Duplicates n-th line of a 2D numpy array n times, and deletes the last line.
    
    args:
        M: A 2D numpy array.
        n: The line to duplicate.
    returns:
        A 2D numpy array with the n-th line duplicated 2 times (at index n and n+1), and the last line deleted.
    """
    M = np.insert(M, n+1, M[n], axis=0)
    M = np.delete(M, -1, axis=0)
    return M


def digit_borders(digit_inside):
    """
    Returns the borders of a digit.
    
    args:
        digit_inside: A 2D numpy array representing the inside of a digit.
    returns:
        A 2D numpy array representing the borders of the digit.
    """
    digit_borders = np.zeros_like(digit_inside)
    for i in range(1, digit_inside.shape[0]-1):
        for j in range(1, digit_inside.shape[1]-1):
            if digit_inside[i,j] == 1:
                if digit_inside[i-1,j] == 0 or digit_inside[i+1,j] == 0 or digit_inside[i,j-1] == 0 or digit_inside[i,j+1] == 0:
                    digit_borders[i,j] = 1
    return digit_borders


def save_rgb_image_plt(image_array, filename):
    """
    Saves an RGB image from a NumPy array of floats between 0 and 1 using Matplotlib.

    Args:
      image_array: A NumPy array of dimension [Nx, Ny, 3] containing the RGB image data as floats between 0 and 1.
      filename: The filename to save the image to.
    """
    # Clip the values between 0 and 1
    image_array = np.clip(image_array, 0, 1)
    # Create a figure with the right dimensions
    fig_loc, ax_loc = plt.subplots(figsize=(image_array.shape[1] / 100, image_array.shape[0] / 100), dpi=100)
    # No need to convert to uint8 for plt.imshow
    ax_loc.imshow(image_array)
    # Hide axes for cleaner image
    ax_loc.axis("off")
    # Save the image
    fig_loc.savefig(filename, bbox_inches="tight", pad_inches=0)
    # Close the plot to free up memory
    plt.close(fig_loc)

def contrast_rgb(M, force):
    """
    Increases the contrast of an RGB image by applying a power-law transformation.
    
    args:
        M: A 3D numpy array representing an RGB image. (Nx, Ny, 3)
        force: The power-law exponent.
    returns:
        A 3D numpy array representing the contrast-enhanced RGB image.
    """
    M = normalize(M)
    M = 0.5 - np.cos(M * np.pi) / 2
    
    # Apply a sigmoid function to increase the contrast
    # M = 1 + np.arctan(M) / np.pi
    
    return M

def matrix_from_black_white_png(png_name):
    """
    Reads a black and white PNG image and returns a 2D numpy array.
    
    args:
        png_name: The name of the PNG file.
    returns:
        A 2D numpy array representing the black and white image and its dimensions.
    """
    # Read the image
    image = plt.imread(png_name)
    # Convert to black and white
    image = np.mean(image, axis=2)
    return image, image.shape[0], image.shape[1]