import numpy as np
from scipy.fft import dctn, idctn

def laplacian2D_periodic(M,dx):
    return (
        - 2 * M + np.roll(M,shift=+1,axis=0) + np.roll(M,shift=-1,axis=0) # second derivative in x
        - 2 * M + np.roll(M,shift=+1,axis=1) + np.roll(M,shift=-1,axis=1) # second derivative in y
    ) / (dx ** 2)
    
def laplacian2D_reflective(M,dx):
    # with reflective boundary conditions
    return (
        - 4 * M 
        + np.pad(M,((1,0),(0,0)),mode='reflect')[:-1,:]     # second derivative in x
        + np.pad(M,((0,1),(0,0)),mode='reflect')[1:,:]      # second derivative in x
        + np.pad(M,((0,0),(1,0)),mode='reflect')[:,:-1]     # second derivative in y
        + np.pad(M,((0,0),(0,1)),mode='reflect')[:,1:]      # second derivative in y
    ) / (dx ** 2)
    
def laplacian2D_zeros_edge(M,dx):
    # with reflective boundary conditions
    return (
        - 4 * M 
        + np.pad(M,((1,0),(0,0)), 'constant', constant_values=0)[:-1,:]     # second derivative in x
        + np.pad(M,((0,1),(0,0)), 'constant', constant_values=0)[1:,:]      # second derivative in x
        + np.pad(M,((0,0),(1,0)), 'constant', constant_values=0)[:,:-1]     # second derivative in y
        + np.pad(M,((0,0),(0,1)), 'constant', constant_values=0)[:,1:]      # second derivative in y
    ) / (dx ** 2)

def draw_ellipse_mask(Lx, Ly, a, b):
    E = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            if (np.abs(Lx//2 - i) <= a and np.abs(Ly//2 - j) <= b) or (
                np.sqrt((a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b) or (
                np.sqrt((-a + Lx//2 - i)**2 + (Ly//2 - j)**2) <= b):
                E[i,j] = 1
    return E

def normalize(M):
    return (M - np.min(M)) / (np.max(M) - np.min(M))

def show_patterns(U, color, ax=None):
    ax.imshow(U, cmap=color,
              interpolation='bilinear')
    ax.set_axis_off()
    
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
    for i in range(Lx):
        for j in range(Ly):
            concentration_dct[i,j] = concentration_dct[i,j] / (1 + 2 * np.pi * diffusivity * dt * ((i/Lx)**2 + (j/Ly)**2))

    # Inverse DCT to get updated concentration field
    updated_grid = idctn(concentration_dct, norm='ortho')

    return updated_grid