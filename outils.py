import numpy as np
from scipy.fft import dctn, idctn
from functools import lru_cache

def draw_ellipse_mask(Lx, Ly, a, b):
    E = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            if (np.abs(Lx//2 - i) <= a and np.abs(Ly//2 - j) <= b) or (
                np.sqrt((a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b) or (
                np.sqrt((-a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b):
                E[i,j] = 1
    return E

def draw_ellipse_mask_2(Lx, Ly, L, W, e):
    E = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            IN_RECTANGLE = (np.abs(Lx//2 - i) < L//2) and (np.abs(Ly//2 - j) < W//2)
            IN_PLUS_ELLIPSE = False
            IN_MINUS_ELLIPSE = False
            if e != 0 and W != 0:
                IN_PLUS_ELLIPSE = (np.sqrt((L//2 + Lx//2 - i)**2/e**2 + (Ly//2 - j)**2/(W//2)**2) < 1)
                IN_MINUS_ELLIPSE = (np.sqrt((-L//2 + Lx//2 - i)**2/e**2 + (Ly//2 - j)**2/(W//2)**2) < 1)
            if IN_RECTANGLE or IN_PLUS_ELLIPSE or IN_MINUS_ELLIPSE:
                E[i,j] = 1
    return E

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