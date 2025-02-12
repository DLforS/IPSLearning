import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange
from joblib import Parallel, delayed

# SDE particle system simulation parameters
N = 20000
nu = 1
T = 1
dt = 1e-4
num_steps = int(T / dt)

# Initial distribution
X = np.concatenate([
    np.random.normal(1, np.sqrt(0.25), N // 2),
    np.random.normal(-1, np.sqrt(0.25), N // 2)
])

@njit
def phi(r):
    return 3 * r**2

# Parallelize force computation using Numba
@njit(parallel=True)
def compute_forces(X):
    N = X.shape[0]
    forces = np.zeros(N)
    for i in prange(N):
        xi = X[i]
        force = 0.0
        for j in range(N):
            if i != j:
                xj = X[j]
                rij = xi - xj
                dist = abs(rij)
                if dist > 1e-12:
                    phi_val = phi(dist)
                    force += phi_val * (rij / dist)
        forces[i] = force
    return forces

# Simulate the system
positions = np.zeros((num_steps, N))
for t in tqdm(range(num_steps), desc="Simulating SDE", ncols=100):
    forces = compute_forces(X)
    X += -forces / N * dt + np.sqrt(2 * nu * dt) * np.random.randn(N)
    positions[t] = X.copy()

# Set up the grid
x_grid = np.linspace(-4, 4, 100)

# Define a function to compute KDE in parallel
def compute_kde_for_t(data, x_grid):
    kde = gaussian_kde(data)
    return kde(x_grid)

# Compute KDE for each time step in parallel
density_list = Parallel(n_jobs=-1, backend='loky')(delayed(compute_kde_for_t)(positions[t], x_grid) for t in tqdm(range(num_steps), desc="Calculating density", ncols=100))
density_matrix = np.array(density_list)

# Save data to CSV
density_df = pd.DataFrame(density_matrix, columns=[f'x_{i}' for i in range(len(x_grid))])
density_df.to_csv('N=2e4.csv', index=False)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(density_matrix.T, extent=[0, T, x_grid.min(), x_grid.max()],
           aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Density')
plt.xlabel('time t')
plt.ylabel('space x')
plt.title('Particle Density Distribution Over Time')
plt.savefig('figure_1.png', dpi=300)
plt.show()