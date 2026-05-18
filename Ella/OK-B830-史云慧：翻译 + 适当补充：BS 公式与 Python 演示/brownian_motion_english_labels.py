# -*- coding: utf-8 -*-
"""
Brownian Motion Visualization
============================

This code generates and visualizes Brownian motion (standard Wiener process) paths.
Brownian motion is fundamental to the Black-Scholes-Merton option pricing model,
representing the random fluctuations in asset prices.

Based on original code from: https://mrislambd.github.io/statandprob/posts/optionprice/
Date: May 14, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Parameter Settings ====================
# Configuration for Brownian motion simulation
n_steps = 100        # Number of time steps
n_paths = 20         # Number of paths
time_horizon = 1     # Time horizon
dt = time_horizon / n_steps  # Time step size
t = np.linspace(0, time_horizon, n_steps)  # Time array

# ==================== Brownian Motion Generation ====================
def generate_brownian_paths(n_paths, n_steps, dt):
    """
    Generate Brownian motion paths

    Mathematical definition of Brownian motion (Wiener process):
    - B(0) = 0
    - Has independent increments
    - For t>s, B(t)-B(s) ~ N(0, t-s)
    - Has continuous paths

    Parameters:
    --------
    n_paths : int
        Number of paths to generate
    n_steps : int
        Number of steps for each path
    dt : float
        Time step size

    Returns:
    -------
    numpy.ndarray
        Array of shape (n_paths, n_steps) containing all paths

    Algorithm:
    ----------
    1. Generate standard normal random increments
    2. Scale increments by sqrt(dt) (since Var(B(t+dt)-B(t)) = dt)
    3. Cumulative sum to obtain paths
    """
    # Generate standard normal random increments
    increments = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

    # Cumulative sum to generate Brownian paths
    brownian_paths = np.cumsum(increments, axis=1)

    return brownian_paths

# ==================== Generate Path Data ====================
# Set random seed for reproducibility
np.random.seed(42)

# Generate single Brownian path (for left plot)
single_path = generate_brownian_paths(1, n_steps, dt)[0]

# Generate multiple Brownian paths (for right plot)
multiple_paths = generate_brownian_paths(n_paths, n_steps, dt)

# ==================== Visualization Setup ====================
# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.9))

# ==================== Left Plot: Single Path ====================
# Plot single Brownian motion path
axes[0].plot(t, single_path, label="Single Path", linewidth=2)
axes[0].set_title("Brownian Motion: Single Path")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Position")
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# ==================== Right Plot: Multiple Paths ====================
# Plot multiple Brownian motion paths
for path in multiple_paths:
    axes[1].plot(t, path, alpha=0.5, linewidth=0.8)

axes[1].set_title(f"Brownian Motion: {n_paths} Paths")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Position")
axes[1].grid(True, alpha=0.3)

# ==================== Figure Optimization ====================
# Adjust subplot spacing
plt.tight_layout()

# Save figure with high quality
plt.savefig('brownian_motion_english.png', dpi=300, bbox_inches='tight')

# Display figure
plt.show()

print("Brownian motion visualization completed! Image saved as 'brownian_motion_english.png'")

# ==================== Mathematical Properties ====================
print("\n=== Mathematical Properties of Brownian Motion ===")
print("1. Initial condition: B(0) = 0")
print("2. Independent increments: For any t1<t2<t3<t4, B(t2)-B(t1) and B(t4)-B(t3) are independent")
print("3. Normal distribution: B(t) ~ N(0, t)")
print("4. Continuous paths: Brownian motion paths are continuous with probability 1")
print("5. Variance property: Var(B(t)) = t")

# ==================== Financial Applications ====================
print("\n=== Applications in Finance ===")
print("In the Black-Scholes-Merton model, asset price S(t) follows geometric Brownian motion:")
print("dS(t) = μS(t)dt + σS(t)dB(t)")
print("where B(t) is standard Brownian motion.")
print("Brownian motion represents the random fluctuation component of asset prices.")