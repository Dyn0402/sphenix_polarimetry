#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 12 12:49 2024
Created in PyCharm
Created as sphenix_polarimetry/hourglass_visualization

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters for the blue curves
    a_blue = 0.5  # Scaling factor for x^2/a^2 term
    b_blue = 0  # Vertical shift for blue
    theta_blue = np.deg2rad(0)  # Rotation angle for blue in degrees
    y_offset_blue = +5  # Vertical offset for the blue curve

    # Parameters for the yellow curves
    a_yellow = 0.5  # Scaling factor for z^2/a^2 term
    b_yellow = -0.  # Vertical shift for yellow
    theta_yellow = np.deg2rad(-0)  # Rotation angle for yellow in degrees
    y_offset_yellow = 0  # Vertical offset for the yellow curve

    # Create the z values
    z = np.linspace(-5, 5, 400)

    # Compute y values for both the blue and yellow curves
    y1_blue = f(z, a_blue, b_blue)
    y2_blue = reflected_f(z, a_blue, b_blue)

    y1_yellow = f(z, a_yellow, b_yellow)
    y2_yellow = reflected_f(z, a_yellow, b_yellow)

    # Rotate and shift the curves (keep z fixed, adjust only y values)
    y1_blue_rot = rotate_curve_fixed_x(z, y1_blue, theta_blue, y_offset_blue)
    y2_blue_rot = rotate_curve_fixed_x(z, y2_blue, theta_blue, y_offset_blue)
    y_blue_avg = (y1_blue_rot + y2_blue_rot) / 2

    y1_yellow_rot = rotate_curve_fixed_x(z, y1_yellow, theta_yellow, y_offset_yellow)
    y2_yellow_rot = rotate_curve_fixed_x(z, y2_yellow, theta_yellow, y_offset_yellow)
    y_yellow_avg = (y1_yellow_rot + y2_yellow_rot) / 2

    fig, ax = plt.subplots()

    # Plot the blue curves
    ax.plot(z, y1_blue_rot, color='blue', label='Blue Curves')
    ax.plot(z, y2_blue_rot, color='blue')
    ax.fill_between(z, y1_blue_rot, y2_blue_rot, color='blue', alpha=0.1)
    ax.plot(z, y_blue_avg, color='blue', linestyle='--', alpha=0.4, label='Blue Average')

    # Plot the yellow curves
    ax.plot(z, y1_yellow_rot, color='orange', label='Yellow Curves')
    ax.plot(z, y2_yellow_rot, color='orange')
    ax.fill_between(z, y1_yellow_rot, y2_yellow_rot, color='orange', alpha=0.3)
    ax.plot(z, y_yellow_avg, color='orange', linestyle='--', alpha=0.4, label='Yellow Average')

    # Add labels and legend
    ax.set_title('Hourglass Visualization')
    ax.set_xlabel('z')
    ax.set_ylabel('x')

    # Display the plot
    ax.grid(True)
    fig.tight_layout()

    overlap = np.minimum(y1_blue_rot, y1_yellow_rot) - np.maximum(y2_blue_rot, y2_yellow_rot)
    fig_overlap, ax_overlap = plt.subplots()
    ax_overlap.plot(z, overlap, color='red', label='Overlap')
    fig_overlap.tight_layout()

    plt.show()

    print('donzo')


# Function for the original curve
def f(x, a, b):
    return np.sqrt(1 + (x ** 2 / a ** 2)) + b


# Function for the reflected curve
def reflected_f(x, a, b):
    return -f(x, a, b)


# Function to rotate y-values while keeping x-values fixed
def rotate_curve_fixed_x(x, y, theta, y_offset=0):
    # Rotate only the y-values, keep x fixed
    y_rot = y * np.cos(theta) - x * np.sin(theta) + y_offset
    return y_rot


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

