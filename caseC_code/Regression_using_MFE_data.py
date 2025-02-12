# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:01:40 2025

@author: 12491
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import eval_legendre
import warnings

warnings.filterwarnings("ignore")  

'''Numerical Solution of MFE'''
# Define parameters
x_min, x_max = -5, 5  
t_min, t_max = 0, 1   
dx = 0.05            
dt = 0.0001          
nu = 1  
nx = int((x_max - x_min) / dx) + 1  
nt = int((t_max - t_min) / dt) + 1  

# Define grid
x = np.linspace(x_min, x_max, nx)
t = np.linspace(t_min, t_max, nt)

# Initial condition 
u = np.zeros((nt, nx))
u[0, :] = 0.5 * (norm.pdf(x, 1, np.sqrt(0.25)) + norm.pdf(x, -1, np.sqrt(0.25)))

# Define true phi function 
def phi(r):
    return 4*r**3 - 4*r

# True K_phi(x) 
def K_phi(x):
    return phi(abs(x)) * np.sign(x)

# Compute convolution term
def convolve_K_phi_u(u_current, x):
    K_phi_values = K_phi(x)
    return fftconvolve(u_current, K_phi_values, mode='same') * dx  

# MFE right-hand side 
def compute_rhs(u_current, x):
    u_xx = (np.roll(u_current, -1) - 2*u_current + np.roll(u_current, 1)) / dx**2  
    nu_uxx = nu * u_xx
    conv_term = convolve_K_phi_u(u_current, x)
    u_conv = u_current * conv_term
    gradient_u_conv = (np.roll(u_conv, -1) - np.roll(u_conv, 1)) / (2*dx)  # Central difference
    return nu_uxx + gradient_u_conv

# RK4 method
for n in range(0, nt-1):
    k1 = dt * compute_rhs(u[n, :], x)
    k2 = dt * compute_rhs(u[n, :] + 0.5*k1, x)
    k3 = dt * compute_rhs(u[n, :] + 0.5*k2, x)
    k4 = dt * compute_rhs(u[n, :] + k3, x)
    u[n+1, :] = u[n, :] + (k1 + 2*k2 + 2*k3 + k4) / 6

# Plot heatmap 
plt.figure(figsize=(8, 6))
plt.imshow(u.T, extent=[t_min, t_max, x_min, x_max], aspect='auto', cmap='hot')
plt.colorbar(label='u(x, t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Heatmap of u(x, t)')
plt.show()

'''psi is the learned phi '''

# Candidate library using Legendre polynomials 
def generate_candidate_library(x, degree=5):        # Maximum degree (greater than or equal to 3)
    # Normalize x to [-1, 1] interval, original x range is [0, 2.5]
    x_normalized = (2 * x / 2.5) - 1
    library = []
    for d in range(degree + 1):
        term = eval_legendre(d, x_normalized)
        library.append(term)
    return np.vstack(library).T  # Shape: (nx, num_terms)

# Compute K_psi(x) = psi(|x|) * sign(x)
def calculate_K_psi(x, xi):
    abs_x = np.abs(x)
    Theta_abs = generate_candidate_library(abs_x, degree=5)
    psi_values = np.dot(Theta_abs, xi)  # psi(|x|) = Theta(|x|) · xi  
    ratio = np.divide(x, abs_x, out=np.zeros_like(x), where=abs_x != 0)  # Handle x=0
    return psi_values * ratio

# Compute error at a single timestep 
def calculate_error_at_timestep(n, u, dt, dx, x, xi):
    u_t = (u[n, :] - u[n-1, :]) / dt  
    u_xx = (np.roll(u[n, :], -1) - 2*u[n, :] + np.roll(u[n, :], 1)) / dx**2
    nu_u_xx = nu * u_xx
    K_psi = calculate_K_psi(x, xi)  
    conv_term = fftconvolve(u[n, :], K_psi, mode='same') * dx
    u_conv = u[n, :] * conv_term
    gradient_u_conv = (np.roll(u_conv, -1) - np.roll(u_conv, 1)) / (2 * dx)
    error = u_t - nu_u_xx - gradient_u_conv
    return np.sum(error**2)

# Compute total error
def calculate_total_error(xi, u, dt, dx, x):
    errors = np.array([calculate_error_at_timestep(n, u, dt, dx, x, xi) for n in range(1, nt)])
    return errors

# Importance sampling optimization 
def importance_sampling_optimization(u, dt, dx, x, num_terms, 
                                     n_iterations=10, sample_size=100, 
                                     threshold=0.01, lambda_reg=1):
    initial_xi = np.random.randn(num_terms)
    xi = initial_xi.copy()
    mask = np.ones_like(xi, dtype=bool)  # Initially all True
    xi_history = [xi.copy()]
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}")
        all_errors = calculate_total_error(xi, u, dt, dx, x)
        total_error = np.sum(all_errors)
        if total_error == 0:
            print("Total error is zero, stopping iteration.")
            break
        sampling_probs = all_errors / total_error
        
        if np.any(np.isnan(sampling_probs)) or np.any(np.isinf(sampling_probs)):
            print("Sampling probabilities contain invalid values, please check error calculation.")
            break
        
        try:
            selected_timesteps = np.random.choice(
                np.arange(1, nt), 
                size=min(sample_size, nt-1), 
                p=sampling_probs,
                replace=False
            )
        except ValueError as e:
            print(f"Error during sampling: {e}")
            break
        
        def sampled_objective(xi_subset):
            full_xi = xi.copy()
            full_xi[mask] = xi_subset
            total_error = 0
            for n in selected_timesteps:
                error = calculate_error_at_timestep(n, u, dt, dx, x, full_xi)
                importance_weight = 1.0 / (sampling_probs[n-1] * sample_size)
                total_error += error * importance_weight
            return total_error + lambda_reg * np.sum(np.abs(xi_subset))
        
        xi_subset_initial = xi[mask]
        result = minimize(sampled_objective, xi_subset_initial, method='L-BFGS-B')
        
        if not result.success:
            print(f"Optimization did not converge: {result.message}")
            break
        
        xi[mask] = result.x
        print("Optimized xi:", xi)
        
        max_abs_coef = np.max(np.abs(xi))
        threshold_value = threshold * max_abs_coef
        small_coef_indices = np.where((np.abs(xi) < threshold_value) & mask)[0]
        print(f"Current threshold: {threshold_value}")
        print("Indices of coefficients to be sparsified:", small_coef_indices)
        
        xi[small_coef_indices] = 0
        mask[small_coef_indices] = False
        xi_history.append(xi.copy())
        
        current_loss = result.fun
        print(f"Current loss value (with sparsity constraint): {current_loss:.6f}")
        
        if not np.any(mask):
            print("All coefficients have been masked. Stopping iteration.")
            break
    
    print("\nFinal optimized xi:", xi)
    return xi, mask, xi_history

# Define candidate library
degree = 5
Theta = generate_candidate_library(x, degree)
num_terms = Theta.shape[1]

# Perform importance sampling optimization
optimized_xi, final_mask, xi_history = importance_sampling_optimization(
    u, dt, dx, x, num_terms,
    n_iterations=10,
    sample_size=100,
    threshold=0.01,
    lambda_reg=1
)

# Full dataset regression
def full_regression_with_mask(xi_initial, mask, u, dt, dx, x, lambda_reg=1):
    def full_objective(xi_subset):
        full_xi = xi_initial.copy()
        full_xi[mask] = xi_subset
        total_error = calculate_total_error(full_xi, u, dt, dx, x)
        return np.sum(total_error)
    
    xi_subset_initial = xi_initial[mask]
    result = minimize(full_objective, xi_subset_initial, method='L-BFGS-B')
    
    if result.success:
        full_xi = xi_initial.copy()
        full_xi[mask] = result.x
        return full_xi
    else:
        print(f"Full dataset regression did not converge: {result.message}")
        return xi_initial

final_xi = full_regression_with_mask(optimized_xi, final_mask, u, dt, dx, x, lambda_reg=1)
total_error_final = calculate_total_error(final_xi, u, dt, dx, x)
final_loss = np.sum(total_error_final)
print(f"Final Loss without regularization: {final_loss:.6f}")

# Plot learned phi(r) vs true phi(r)
def learned_phi(r_values, xi):
    library = generate_candidate_library(r_values, degree=5)
    phi_values = np.dot(library, xi)
    return phi_values

r_plot = np.linspace(0, 2.5, 100)
phi_true = phi(r_plot)
phi_learned = learned_phi(r_plot, final_xi)

plt.figure(figsize=(8, 6))
plt.plot(r_plot, phi_true, label='True $\phi(r)$', linestyle='-', color='blue')
plt.plot(r_plot, phi_learned, label='Learned $\phi(r)$', linestyle='--', color='red')
plt.xlabel('Radius $r$')
plt.ylabel(r'$\phi(r)$')
plt.title('Comparison of True and Learned $\phi(r)$')
plt.legend()
plt.grid(True)
plt.show()