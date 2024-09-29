#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 11 12:24 2024
Created in PyCharm
Created as sphenix_polarimetry/beam_beam_sim

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit as cf
from PIL import Image, ImageSequence
from time import sleep

from BunchDensity import BunchDensity


def main():
    # animate_bunch_density_propagation()
    # animate_bunch_collision()
    # simulate_vernier_scan()
    hourglass_head_on_z_dist_comparison()
    print('donzo')


def hourglass_head_on_z_dist_comparison():
    """
    Plot the z-vertex distribution for head-on collisions with and without the hourglass effect.
    """
    mu = '\u03BC'
    # Initialize two bunches
    bunch1 = BunchDensity()
    bunch2 = BunchDensity()
    beta_star_off = None  # cm
    beta_star_on = 85  # cm

    # x_offset = 0.  # microns  Head on
    x_offset = 700.  # microns  Peripheral

    # Set initial positions, velocities, and widths for the two bunches
    bunch1.set_initial_z(-600.e4)  # um Initial z position of bunch 1
    bunch1.set_offsets(x_offset, 0.)  # um Initial x and y offsets of bunch 1
    bunch1.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch 1 moving in +z direction
    bunch1.set_sigma(170., 170., 1.1e6)  # um Width of bunch 1 in x, y, z
    bunch1.set_angles(-0.0e-3, 0.0)  # rad Rotate bunch 1 in the x-z and y-z planes
    bunch1.beta_star = beta_star_off

    bunch2.set_initial_z(600.e4)  # um Initial z position of bunch 2
    bunch2.set_offsets(0., 0.)  # um Initial x and y offsets of bunch 2
    bunch2.set_beta(0., 0., -1.)  # Dimensionless velocity of bunch 2 moving in -z direction
    bunch2.set_sigma(170., 170., 1.1e6)  # um Width of bunch 2 in x, y, z
    bunch2.set_angles(-0.0e-3, 0.0)  # rad Rotate bunch 2 in the x-z and y-z planes
    bunch2.beta_star = beta_star_off

    n_propagation_points = 50
    n_density_points = 101
    xy_lim_sigma = 10
    z_lim_size = 1.2

    # Set timestep for propagation
    bunch1.dt = bunch2.dt = (bunch2.initial_z - bunch1.initial_z) / bunch1.c / n_propagation_points

    # Create a grid of points for the x-z and y-z planes
    x = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[0], xy_lim_sigma * bunch1.transverse_sigma[0], n_density_points)
    y = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[1], xy_lim_sigma * bunch1.transverse_sigma[1], n_density_points)
    z = np.linspace(-2.5e6, 2.5e6, n_density_points + 5)

    X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z, indexing='ij')  # For 3D space

    z_cm = z / 1e4

    integrated_density_product_z_hg_off = np.zeros_like(z)
    for i in range(n_propagation_points):
        print(f'Beta Star Off Propagation {i}/{n_propagation_points}')
        density1_xyz = bunch1.density(X_3d, Y_3d, Z_3d)
        density2_xyz = bunch2.density(X_3d, Y_3d, Z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        # Sum over x and y to get the density product in z
        integrated_density_product_z_hg_off += np.sum(density_product_xyz, axis=(0, 1))

        bunch1.propagate()
        bunch2.propagate()

    integrated_density_product_z_hg_off /= n_propagation_points

    # Set initial positions, velocities, and widths for the two bunches
    bunch1.set_initial_z(-600.e4)  # um Initial z position of bunch 1
    bunch1.set_offsets(x_offset, 0.)  # um Initial x and y offsets of bunch 1
    bunch1.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch 1 moving in +z direction
    bunch1.set_sigma(170., 170., 1.1e6)  # um Width of bunch 1 in x, y, z
    bunch1.set_angles(-0.0e-3, 0.0)  # rad Rotate bunch 1 in the x-z and y-z planes
    bunch1.set_beta_star(beta_star_on)

    bunch2.set_initial_z(600.e4)  # um Initial z position of bunch 2
    bunch2.set_offsets(0., 0.)  # um Initial x and y offsets of bunch 2
    bunch2.set_beta(0., 0., -1.)  # Dimensionless velocity of bunch 2 moving in -z direction
    bunch2.set_sigma(170., 170., 1.1e6)  # um Width of bunch 2 in x, y, z
    bunch2.set_angles(-0.0e-3, 0.0)  # rad Rotate bunch 2 in the x-z and y-z planes
    bunch2.set_beta_star(beta_star_on)

    integrated_density_product_z_hg_on = np.zeros_like(z)
    for i in range(n_propagation_points):
        print(f'Beta Star On Propagation {i}/{n_propagation_points}')
        density1_xyz = bunch1.density(X_3d, Y_3d, Z_3d)
        density2_xyz = bunch2.density(X_3d, Y_3d, Z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        # Sum over x and y to get the density product in z
        integrated_density_product_z_hg_on += np.sum(density_product_xyz, axis=(0, 1))

        bunch1.propagate()
        bunch2.propagate()

    integrated_density_product_z_hg_on /= n_propagation_points

    fig, ax = plt.subplots(figsize=(7, 3), dpi=144)
    p0 = [max(integrated_density_product_z_hg_off), 1.e6, 0.]
    print(f'Initial guess: {p0}')
    popt_off, pcov_off = cf(gaus_1d, z, integrated_density_product_z_hg_off, p0=p0)
    popt_on, pcov_on = cf(gaus_1d, z, integrated_density_product_z_hg_on, p0=p0)
    ax.plot(z_cm, integrated_density_product_z_hg_off, color='black', label='Hourglass Effect Off')
    ax.plot(z_cm, integrated_density_product_z_hg_on, color='red', label='Hourglass Effect On')
    integral_ratio = np.sum(integrated_density_product_z_hg_on) / np.sum(integrated_density_product_z_hg_off)
    width_ratio = popt_on[1] / popt_off[1]
    width_percent = (1 - width_ratio) * 100
    offset_str = fr'Offset: {x_offset:.0f}{mu}m'
    if integral_ratio < 1:
        integral_percent = (1 - integral_ratio) * 100
        percent_str = f'{integral_percent:.0f}% fewer collisions'
    else:
        integral_percent = (integral_ratio - 1) * 100
        percent_str = f'{integral_percent:.0f}% more collisions'
    ax.annotate(f'{offset_str}\n{percent_str}\nwith hourglass effect', (0.03, 0.95),
                xycoords='axes fraction', va='top', ha='left', bbox=dict(facecolor='wheat', alpha=0.3))

    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Integrated Density Product')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.145)

    plt.show()


def simulate_vernier_scan():
    x_offsets = np.linspace(-1., 1., 9) * 1e3  # mm to microns

    beta_star_actual = 85  # cm
    beta_star_off = 9e11  # cm Effectively turned off
    # beta_star = 9e11  # cm
    bunch_lengths = [1.3 * 1e6, 1.17 * 1e6]  # m to microns
    # bunch_length = 1.2 * 1e6  # m to microns
    bunch_width = 135.  # microns Transverse Gaussian bunch width
    # bunch_width = 200.  # microns Transverse Gaussian bunch width
    z_initial = 6.e6  # microns Starting z position for center of bunches
    # crossing_angle = 1.5e-3 / 2  # radians
    # crossing_angle = 1.5e-4 / 2  # radians
    crossing_angle = 0.08e-3 / 2  # radians
    crossing_angle = 0.0  # radians

    n_propagation_points = 50
    n_density_points = 101
    xy_lim_sigma = 10
    z_lim_sigma = 7

    beta_stars = {'Hourglass Effect On': beta_star_actual, 'Hourglass Effect Off': beta_star_off}

    beta_star_amps, beta_star_sigmas = [], []
    for beta_star_name, beta_star in beta_stars.items():
        print(f'Starting {beta_star_name}')
        prob_density_product_sum = []
        for x_offset in x_offsets:
            print(f'x_offset: {x_offset} microns')

            # Initialize two bunches
            bunch1 = BunchDensity()
            bunch2 = BunchDensity()

            # Set initial positions, velocities, and widths for the two bunches
            # bunch1.set_r(0., 0., -10.e6)  # μm Initial position of bunch 1
            bunch1.set_r(x_offset, 0., -z_initial)  # μm Initial position of bunch 1
            bunch1.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch 1 moving in +z direction
            bunch1.set_sigma(bunch_width, bunch_width, bunch_lengths[0])  # μm Width of bunch 1 in x, y, z
            bunch1.set_angle(-crossing_angle)  # Rotate bunch 1 in the y-z plane
            bunch1.beta_star = beta_star

            # bunch2.set_r(0., 0., 10.e6)  # μm Initial position of bunch 2
            bunch2.set_r(0., 0., z_initial)  # μm Initial position of bunch 2
            bunch2.set_beta(0., 0., -1.)  # Dimensionless velocity of bunch 2 moving in -z direction
            bunch2.set_sigma(bunch_width, bunch_width, bunch_lengths[1])  # μm Width of bunch 2 in x, y, z
            bunch2.set_angle(crossing_angle)  # Rotate bunch 2 in the y-z plane
            bunch2.beta_star = beta_star

            # Set timestep for propagation
            # bunch1.dt = bunch2.dt = 1e-2  # ns Timestep to propagate both bunches
            bunch1.dt = bunch2.dt = (bunch2.r[2] - bunch1.r[
                2]) / bunch1.c / n_propagation_points  # ns Timestep to propagate both bunches

            # Create a grid of points for the x-z and y-z planes
            x = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[0], xy_lim_sigma * bunch1.transverse_sigma[0], n_density_points)
            y = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[1], xy_lim_sigma * bunch1.transverse_sigma[1], n_density_points)
            z = np.linspace(-z_lim_sigma * bunch1.transverse_sigma[2], z_lim_sigma * bunch1.transverse_sigma[2], n_density_points + 5)

            X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z, indexing='ij')  # For 3D space

            prob_density_product_sum_i = 0
            for i in range(n_propagation_points):
                density1_xyz = bunch1.density(X_3d, Y_3d, Z_3d)
                density2_xyz = bunch2.density(X_3d, Y_3d, Z_3d)

                # Calculate the density product
                density_product_xyz = density1_xyz * density2_xyz

                prob_density_product_sum_i += np.sum(density_product_xyz)

                bunch1.propagate()
                bunch2.propagate()

            prob_density_product_sum.append(prob_density_product_sum_i / n_propagation_points)

        p0 = [max(prob_density_product_sum), 150, 0.]
        popt, pcov = cf(vernier_scan_fit, x_offsets, prob_density_product_sum, p0=p0)
        popt[1] = abs(popt[1])  # Sigma is squared, so could be negative
        beta_star_amps.append(popt[0])
        beta_star_sigmas.append(popt[1])
        perr = np.sqrt(np.diag(pcov))
        fig, ax = plt.subplots()
        ax.axhline(0, color='black', alpha=0.5)
        ax.scatter(x_offsets, prob_density_product_sum, marker='o', color='b', label='Data', zorder=10)
        x_fit = np.linspace(x_offsets.min(), x_offsets.max(), 1000)
        ax.plot(x_fit, vernier_scan_fit(x_fit, *p0), 'gray', alpha=0.5, label='Guess')
        ax.plot(x_fit, vernier_scan_fit(x_fit, *popt), 'r-', label='Fit')
        fit_str = f'Amp = {popt[0]:.1e}±{perr[0]:.1e} \nSigma = {popt[1]:.1f}±{perr[1]:.1f} μm \nMean = {popt[2]:.1f}±{perr[2]:.1f} μm'
        ax.annotate(f'{fit_str}', (0.02, 0.82), xycoords='axes fraction',
                    bbox=dict(facecolor='wheat', alpha=0.5))
        ax.set_xlabel('x Offset (μm)')
        ax.set_ylabel('Integrated Density Product')
        ax.set_title(f'Integrated Density Product vs x Offset {beta_star_name}')
        ax.legend()
        fig.tight_layout()

    # Neglecting hourglass overestimates luminosity. Need to multiply by hourglass on / off, smaller than 1
    beta_star_amp_ratio = beta_star_amps[0] / beta_star_amps[1]  # On / Off

    # Neglecting hourglass overestimates σ, therefore underestimates lumi. Need to multiply by (on / off)**2, > 1
    beta_star_sigma_ratio = beta_star_sigmas[0] / beta_star_sigmas[1]  # On / Off

    lumi_correction = beta_star_amp_ratio * beta_star_sigma_ratio ** 2

    print(f'Ratio of beta star amps: {beta_star_amp_ratio:.2f}')
    print(f'Ratio of beta star sigmas: {beta_star_sigma_ratio:.2f}')
    print(f'Luminosity Correction: {lumi_correction}')
    print(f'Percent Correction: {(lumi_correction - 1) * 100:.1f}%')
    plt.show()


def animate_bunch_collision():
    scan_date = 'Aug12'
    fit_path = f'C:/Users/Dylan/Desktop/vernier_scan/CAD_Measurements/VernierScan_{scan_date}_COLOR_longitudinal_fit.dat'

    # Initialize two bunches
    bunch1 = BunchDensity()
    bunch2 = BunchDensity()
    beta_star = 85  # cm

    # Set initial positions, velocities, and widths for the two bunches
    bunch1.set_initial_z(-600.e4)  # um Initial z position of bunch 1
    bunch1.set_offsets(+900., 0.)  # um Initial x and y offsets of bunch 1
    bunch1.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch 1 moving in +z direction
    bunch1.set_sigma(150., 150., 1.3e6)  # um Width of bunch 1 in x, y, z
    bunch1.set_angles(-0.075e-3, 0.0)  # rad Rotate bunch 1 in the x-z and y-z planes
    bunch1.beta_star = beta_star
    bunch1.read_longitudinal_beam_profile_fit_parameters_from_file(fit_path.replace('_COLOR_', '_blue_'))

    bunch2.set_initial_z(600.e4)  # um Initial z position of bunch 2
    bunch2.set_offsets(0., 0.)  # um Initial x and y offsets of bunch 2
    bunch2.set_beta(0., 0., -1.)  # Dimensionless velocity of bunch 2 moving in -z direction
    bunch2.set_sigma(150., 150., 1.3e6)  # um Width of bunch 2 in x, y, z
    # bunch2.set_angles(-0.115e-3, 0.0)  # rad Rotate bunch 2 in the x-z and y-z planes
    bunch2.beta_star = beta_star
    bunch2.read_longitudinal_beam_profile_fit_parameters_from_file(fit_path.replace('_COLOR_', '_yellow_'))

    n_propagation_points = 50
    n_density_points = 101
    xy_lim_sigma = 10
    z_lim_sigma = 7

    plot_z_gaus_fit = False

    animation_save_name = 'test.gif'

    # Set timestep for propagation
    bunch1.dt = bunch2.dt = (bunch2.initial_z - bunch1.initial_z) / bunch1.c / n_propagation_points
    max_bunch1_density = 1. / (2 * np.pi * bunch1.transverse_sigma[0] * bunch1.transverse_sigma[1] * bunch1.longitudinal_params['sigma1'])
    max_bunch2_density = 1. / (2 * np.pi * bunch2.transverse_sigma[0] * bunch2.transverse_sigma[1] * bunch2.longitudinal_params['sigma1'])
    max_bunch_density_sum = max_bunch1_density + max_bunch2_density
    max_bunch_density_product = max_bunch1_density * max_bunch2_density

    # Create a grid of points for the x-z and y-z planes
    x = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[0], xy_lim_sigma * bunch1.transverse_sigma[0], n_density_points)
    y = np.linspace(-xy_lim_sigma * bunch1.transverse_sigma[1], xy_lim_sigma * bunch1.transverse_sigma[1], n_density_points)
    z = np.linspace(-z_lim_sigma * bunch1.longitudinal_params['sigma1'], z_lim_sigma * bunch1.longitudinal_params['sigma1'], n_density_points + 5)

    z_cm = z / 1e4

    X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z, indexing='ij')  # For 3D space

    # Set up the figure with a wider aspect ratio
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))  # 2 rows, 2 columns, wide figure

    # Density Product visualization (left column)
    im_xz = ax[0, 0].imshow(np.zeros((x.size, z.size)), extent=[z_cm.min(), z_cm.max(), x.min(), x.max()],
                            origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_product, aspect='auto')
    im_yz = ax[1, 0].imshow(np.zeros((y.size, z.size)), extent=[z_cm.min(), z_cm.max(), y.min(), y.max()],
                            origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_product, aspect='auto')

    ax[0, 0].set_title('Density Product in x-z and y-z Planes')
    ax[0, 0].set_xlabel('z (cm)')
    ax[0, 0].set_ylabel('x (μm)')
    ax[1, 0].set_xlabel('z (cm)')
    ax[1, 0].set_ylabel('y (μm)')

    # Density Sum visualization (right column)
    im_sum_xz = ax[0, 1].imshow(np.zeros((x.size, z.size)), extent=[z_cm.min(), z_cm.max(), x.min(), x.max()],
                                origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_sum, aspect='auto')
    im_sum_yz = ax[1, 1].imshow(np.zeros((y.size, z.size)), extent=[z_cm.min(), z_cm.max(), y.min(), y.max()],
                                origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_sum, aspect='auto')

    ax[0, 1].set_title('Density Sum in x-z and y-z Planes')
    ax[0, 1].set_xlabel('z (cm)')
    # Suppress the y-axis labels and ticks for the right column
    ax[0, 1].set_ylabel('')
    ax[0, 1].set_yticks([])

    ax[1, 1].set_xlabel('z (cm)')
    ax[1, 1].set_ylabel('')
    ax[1, 1].set_yticks([])

    # Adjust layout: no space between plots horizontally or vertically
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    integrated_density_product_z = np.zeros_like(z)
    fig_z_slices, ax_z_slices = plt.subplots(2, 1)

    first_pass = True
    im_xzs, im_yzs, im_sum_xzs, im_sum_yzs = [], [], [], []
    for i in range(n_propagation_points):
        density1_xyz = bunch1.density(X_3d, Y_3d, Z_3d)
        density2_xyz = bunch2.density(X_3d, Y_3d, Z_3d)
        # density1_xyz = bunch1.density_py(X_3d, Y_3d, Z_3d)
        # density2_xyz = bunch2.density_py(X_3d, Y_3d, Z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        # Update the product density plots
        density_product_xz = np.sum(density_product_xyz, axis=1)
        density_product_yz = np.sum(density_product_xyz, axis=0)
        im_xzs.append(density_product_xz)
        im_yzs.append(density_product_yz)
        im_xz.set_data(density_product_xz)
        im_yz.set_data(density_product_yz)

        # Sum over x and y to get the density product in z
        integrated_density_product_z += np.sum(density_product_xyz, axis=(0, 1))
        ax_z_slices[0].plot(z_cm, np.sum(density_product_xyz, axis=(0, 1)), label=f'Prop {i}')
        ax_z_slices[1].plot(z_cm, np.sum(density_product_xyz, axis=(0, 1)), label=f'Prop {i}')

        # Calculate the density sum
        density_sum_xz = np.sum(density1_xyz, axis=1) + np.sum(density2_xyz, axis=1)
        density_sum_yz = np.sum(density1_xyz, axis=0) + np.sum(density2_xyz, axis=0)

        # Update the sum density plots
        im_sum_xzs.append(density_sum_xz)
        im_sum_yzs.append(density_sum_yz)
        im_sum_xz.set_data(density_sum_xz)
        im_sum_yz.set_data(density_sum_yz)

        fig.canvas.draw()
        # fig_sum.canvas.draw()
        plt.pause(0.01)
        if first_pass:
            sleep(5)
            first_pass = False
        bunch1.propagate()
        bunch2.propagate()

    integrated_density_product_z /= n_propagation_points

    ax_z_slices[0].set_title('Integrated Density Product vs z')
    ax_z_slices[0].set_xlabel('z (cm)')
    ax_z_slices[0].set_ylabel('Integrated Density Product')
    ax_z_slices[0].legend()
    ax_z_slices[1].set_title('Integrated Density Product vs z')
    ax_z_slices[1].set_xlabel('z (cm)')
    ax_z_slices[1].set_ylabel('Integrated Density Product')
    ax_z_slices[1].legend()
    fig_z_slices.tight_layout()

    plt.figure()
    p0 = [max(integrated_density_product_z), 1.e6, 0.]
    print(f'Initial guess: {p0}')
    popt, pcov = cf(gaus_1d, z, integrated_density_product_z, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    plt.axhline(0, color='black', alpha=0.5)
    plt.plot(z_cm, integrated_density_product_z)
    if plot_z_gaus_fit:
        z_plot = np.linspace(z.min(), z.max(), 1000)
        plt.plot(z_plot / 1e4, gaus_1d(z_plot, *p0), color='gray', alpha=0.3, label='Guess')
        plt.plot(z_plot / 1e4, gaus_1d(z_plot, *popt), 'r--', label='Fit')
        plt.legend()
    fit_str = f'Amp = {popt[0]:.2e}±{perr[0]:.2e} cm \nSigma = {popt[1] / 1e4:.2f}±{perr[1] / 1e4:.2f} cm'
    sum_str = f'Sum = {np.sum(integrated_density_product_z):.2e}'
    plt.annotate(f'{fit_str}\n{sum_str}', (0.05, 0.85), xycoords='axes fraction')
    plt.xlabel('z (cm)')
    plt.ylabel('Integrated Density Product')
    plt.title('Integrated Density Product vs z')
    plt.tight_layout()
    print(fit_str)
    print(sum_str)

    anim = FuncAnimation(fig, animate_update, frames=int(n_propagation_points * 1.5),
                         fargs=(im_xz, im_yz, im_sum_xz, im_sum_yz, im_xzs, im_yzs, im_sum_xzs, im_sum_yzs))
    anim.save(animation_save_name, writer='pillow', fps=5)
    # set_gif_no_loop(animation_save_name, animation_save_name)

    plt.show()


def animate_update(i, im_xz, im_yz, im_sum_xz, im_sum_yz, im_xzs, im_yzs, im_sum_xzs, im_sum_yzs):
    if i >= len(im_xzs):
        i = len(im_xzs) - 1
    im_xz.set_data(im_xzs[i])
    im_yz.set_data(im_yzs[i])
    im_sum_xz.set_data(im_sum_xzs[i])
    im_sum_yz.set_data(im_sum_yzs[i])


def set_gif_no_loop(input_gif_path, output_gif_path):
    with Image.open(input_gif_path) as img:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
        img_info = img.info

        # Save the GIF with loop count set to 1 (play once)
        frames[0].save(output_gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       loop=1)  # Set loop to 1 to play only once


def animate_bunch_density_propagation():
    bunch = BunchDensity()
    bunch.set_initial_z(-9.e6)  # um Initial position of bunch
    # bunch.set_offsets(100., 0.)  # um Offset of bunch in x, y
    bunch.set_offsets(0., 0.)  # um Offset of bunch in x, y
    # bunch.set_angles(0., 1.e-4)
    bunch.set_angles(1.e-4, 0.)
    # bunch.set_angles(0., 0.)
    bunch.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch
    bunch.set_sigma(100., 100., 1.e6)  # um Width of bunch in x, y, z
    bunch.dt = 1e-1  # ns Timestep to propagate bunch
    bunch.calculate_r_and_beta()
    print(f'Initial r: {bunch.r}')
    print(f'Initial beta: {bunch.beta}')

    n_propagation_points = 600
    n_density_points_x, n_density_points_y, n_density_points_z = 100, 110, 150
    nsigma_x = 15
    nsigma_y = 15
    nsigma_z = 15

    # Create a grid of points for the x-z and y-z planes
    x = np.linspace(-nsigma_x * bunch.transverse_sigma[0], nsigma_x * bunch.transverse_sigma[0], n_density_points_x)
    y = np.linspace(-nsigma_y * bunch.transverse_sigma[1], nsigma_y * bunch.transverse_sigma[1], n_density_points_y)
    z = np.linspace(-nsigma_z * bunch.transverse_sigma[2], nsigma_z * bunch.transverse_sigma[2], n_density_points_z)

    X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z, indexing='ij')  # For 3D space

    max_bunch_density = 1. / ((2 * np.pi) ** 1.5 * bunch.transverse_sigma[0] * bunch.transverse_sigma[1] * bunch.transverse_sigma[2])

    fig, ax = plt.subplots(3, 1, figsize=(15, 8))
    im_xy = ax[0].imshow(np.zeros((n_density_points_y, n_density_points_x)), extent=[y.min(), y.max(), x.min(), x.max()],
                         origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density, aspect='auto')
    im_xz = ax[1].imshow(np.zeros((n_density_points_z, n_density_points_x)), extent=[z.min(), z.max(), x.min(), x.max()],
                         origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density, aspect='auto')
    im_yz = ax[2].imshow(np.zeros((n_density_points_z, n_density_points_y)), extent=[z.min(), z.max(), y.min(), y.max()],
                         origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density, aspect='auto')

    ax[0].set_title('Bunch Propagation in x-y Plane')
    ax[1].set_title('Bunch Propagation in x-z Plane')
    ax[2].set_title('Bunch Propagation in y-z Plane')
    ax[0].set_xlabel('y (um)')
    ax[0].set_ylabel('x (um)')
    ax[1].set_xlabel('z (um)')
    ax[1].set_ylabel('x (um)')
    ax[2].set_xlabel('z (um)')
    ax[2].set_ylabel('y (um)')

    fig.tight_layout()

    for i in range(n_propagation_points):
        print(f'Propagating {i + 1}/{n_propagation_points}: r={bunch.r}')
        density_xyz = bunch.density_py(X_3d, Y_3d, Z_3d)
        density_xy = np.sum(density_xyz, axis=2)
        density_xz = np.sum(density_xyz, axis=1)
        density_yz = np.sum(density_xyz, axis=0)

        print(f'Shape yx: {density_xy.shape}, xz: {density_xz.shape}, yz: {density_yz.shape}, xyz: {density_xyz.shape}')

        # Print index of max density
        max_density_index = np.unravel_index(np.argmax(density_xyz, axis=None), density_xyz.shape)
        print(f'Max density: {np.max(density_xyz)} at {max_density_index}')

        im_xy.set_data(density_xy)
        im_xz.set_data(density_xz)
        im_yz.set_data(density_yz)

        plt.pause(0.01)
        bunch.propagate()

    plt.show()


def gaus_1d(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def vernier_scan_fit(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))


if __name__ == '__main__':
    main()
