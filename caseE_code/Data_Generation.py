# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:03:44 2025

@author: 12491
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange
from joblib import Parallel, delayed

# Set random seed for reproducibility
np.random.seed(42)

########################################################
# Step 1: Define the Interaction Function phi(r)
########################################################

@njit
def f_true(r):
    """
    Compute the piecewise function f(r).
    
    Parameters:
        r (float): Distance value, non-negative.
    
    Returns:
        float: Corresponding f(r) value.
    """
    if 0 <= r <= 1.5:
        return 0.5
    elif 1.5 < r <= 2.0:
        return 1.0
    else:
        return 0.0

@njit
def phi_true_func(r):
    """
    Compute phi(r) = f(r) * r.
    
    Parameters:
        r (float): Distance value, non-negative.
    
    Returns:
        float: Corresponding phi(r) value.
    """
    return f_true(r) * r

########################################################
# Step 2: SDE Particle System Simulation Parameters
########################################################

# Number of particles
N = 20000

# Diffusion coefficient
nu = 0.01

# Total simulation time
T = 10.0

# Time step
dt = 1e-4

# Number of time steps
num_steps = int(T / dt)

# Initialize particle positions
# Combining multiple Gaussian distributions as in the MFE initial condition
half_N = N // 4
X = np.concatenate([
    np.random.normal(-3, np.sqrt(0.64), half_N),
    np.random.normal(0.5, np.sqrt(0.25), half_N),
    np.random.normal(2, np.sqrt(0.81), half_N),
    np.random.normal(3, np.sqrt(0.25), half_N)
])

########################################################
# Step 3: Compute Forces Between Particles
########################################################

@njit(parallel=True)
def compute_forces(X):
    """
    Compute the interaction forces on each particle.
    
    Parameters:
        X (numpy.ndarray): Current positions of particles.
    
    Returns:
        numpy.ndarray: Forces on each particle.
    """
    N = X.shape[0]
    forces = np.zeros(N)
    for i in prange(N):
        xi = X[i]
        force = 0.0
        for j in range(N):
            if i != j:
                xj = X[j]
                rij = xi - xj
                dist = np.abs(rij)
                if dist > 1e-12:
                    phi_val = phi_true_func(dist)
                    force += phi_val * (rij / dist)
        forces[i] = force
    return forces

########################################################
# Step 4: Simulate the SDE System
########################################################

# To store particle positions at each time step
# To save memory, you might consider storing at intervals
# Here, we store every step for higher resolution
positions = np.zeros((num_steps, N), dtype=np.float32)

# Progress bar for simulation
for t in tqdm(range(num_steps), desc="Simulating SDE", ncols=100):
    forces = compute_forces(X)
    # Euler-Maruyama update
    X += (-forces / N) * dt + np.sqrt(2 * nu * dt) * np.random.randn(N)
    positions[t] = X.copy()

########################################################
# Step 5: Perform KDE at Each Time Step
########################################################

# Define the spatial grid for KDE
x_grid = np.linspace(-5, 5, 200)  # Adjust grid range and resolution as needed

# Function to compute KDE for a single time step
def compute_kde_for_t(data, x_grid):
    kde = gaussian_kde(data)
    return kde(x_grid)

# Use parallel processing to speed up KDE computations
density_list = Parallel(n_jobs=-1, backend='loky')(
    delayed(compute_kde_for_t)(positions[t], x_grid) for t in tqdm(range(num_steps), desc="Calculating KDE", ncols=100)
)

# Convert list to numpy array for easier handling
density_matrix = np.array(density_list)

########################################################
# Step 6: Visualize the Density Heatmap
########################################################

# Define time axis for visualization
time_axis = np.linspace(0, T, num_steps)

# Create a meshgrid for plotting
T_mesh, X_mesh = np.meshgrid(time_axis, x_grid)

# Plot the heatmap
plt.figure(figsize=(12, 6))
plt.imshow(density_matrix.T, extent=[0, T, x_grid.min(), x_grid.max()],
           aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Density')
plt.xlabel('Time t')
plt.ylabel('Space x')
plt.title('Particle Density Distribution Over Time (SDE Simulation)')
plt.tight_layout()
plt.show()

########################################################
# Optional: Save Density Data to CSV
########################################################

# If you wish to save the density data for further analysis
density_df = pd.DataFrame(density_matrix, columns=[f'x_{i}' for i in range(len(x_grid))])
density_df.to_csv('jubu_N=2e4.csv', index=False)