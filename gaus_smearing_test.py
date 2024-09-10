#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 10 3:26 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/gaus_smearing_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Generate an ideal position distribution (for example, a delta function or a uniform distribution)
x = np.linspace(-2, 2, 1000)  # x-axis with 1000 points from -10 to 10
ideal_distribution = np.exp(-x**2 / 0.1)

# Define the desired sigma in x-axis units
sigma_x = 0.1  # In units of your x-axis

# Calculate the spacing between x-axis points
dx = x[1] - x[0]  # Assuming uniform spacing

# Scale sigma to index units
sigma_index = sigma_x / dx

# Apply Gaussian smearing with the scaled sigma
smeared_distribution = gaussian_filter1d(ideal_distribution, sigma_index)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, ideal_distribution, label='Ideal Distribution', linestyle='--')
plt.plot(x, smeared_distribution, label='Smeared Distribution', linewidth=2)
plt.legend()
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.title('Gaussian Smearing of Ideal Distribution')
plt.show()
