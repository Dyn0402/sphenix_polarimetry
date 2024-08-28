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
from multiprocessing import Pool

import bunch_density_cpp as bdcpp


def main():
    # animate_bunch_density_propagation()
    animate_bunch_collision()
    # simulate_vernier_scan()
    print('donzo')


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
    crossing_angle = 0.  # radians

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
            x = np.linspace(-xy_lim_sigma * bunch1.sigma[0], xy_lim_sigma * bunch1.sigma[0], n_density_points)
            y = np.linspace(-xy_lim_sigma * bunch1.sigma[1], xy_lim_sigma * bunch1.sigma[1], n_density_points)
            z = np.linspace(-z_lim_sigma * bunch1.sigma[2], z_lim_sigma * bunch1.sigma[2], n_density_points + 5)

            X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z)  # For 3D space

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
    # Initialize two bunches
    bunch1 = BunchDensity()
    bunch2 = BunchDensity()
    beta_star = 85  # cm

    # Set initial positions, velocities, and widths for the two bunches
    # bunch1.set_r(0., 0., -10.e6)  # um Initial position of bunch 1
    bunch1.set_r(0., 0., -6.e6)  # um Initial position of bunch 1
    bunch1.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch 1 moving in +z direction
    # bunch1.set_sigma(150., 150., 4 * bunch1.c)  # um Width of bunch 1 in x, y, z
    bunch1.set_sigma(150., 150., 1.3e6)  # um Width of bunch 1 in x, y, z
    # bunch1.set_angle(-1.5e-3 / 2)  # Rotate bunch 1 in the y-z plane
    bunch1.set_angle(-2e-4 / 2)  # Rotate bunch 1 in the y-z plane
    bunch1.beta_star = beta_star

    # bunch2.set_r(0., 0., 10.e6)  # um Initial position of bunch 2
    bunch2.set_r(0., 0., 6.e6)  # um Initial position of bunch 2
    bunch2.set_beta(0., 0., -1.)  # Dimensionless velocity of bunch 2 moving in -z direction
    # bunch2.set_sigma(150., 150., 4 * bunch1.c)  # um Width of bunch 2 in x, y, z
    bunch2.set_sigma(150., 150., 1.3e6)  # um Width of bunch 2 in x, y, z
    # bunch2.set_angle(1.5e-3 / 2)  # Rotate bunch 2 in the y-z plane
    # bunch2.set_angle(1.e-4 / 2)  # Rotate bunch 2 in the y-z plane
    bunch2.beta_star = beta_star

    n_propagation_points = 50
    n_density_points = 101
    xy_lim_sigma = 10
    z_lim_sigma = 7

    plot_z_gaus_fit = False

    animation_save_name = 'test.gif'

    # Set timestep for propagation
    # bunch1.dt = bunch2.dt = 1e-2  # ns Timestep to propagate both bunches
    bunch1.dt = bunch2.dt = (bunch2.r[2] - bunch1.r[
        2]) / bunch1.c / n_propagation_points  # ns Timestep to propagate both bunches
    max_bunch1_density = 1. / (2 * np.pi * bunch1.sigma[0] * bunch1.sigma[1] * bunch1.sigma[2])
    max_bunch2_density = 1. / (2 * np.pi * bunch2.sigma[0] * bunch2.sigma[1] * bunch2.sigma[2])
    max_bunch_density_sum = max_bunch1_density + max_bunch2_density
    max_bunch_density_product = max_bunch1_density * max_bunch2_density

    # Create a grid of points for the x-z and y-z planes
    x = np.linspace(-xy_lim_sigma * bunch1.sigma[0], xy_lim_sigma * bunch1.sigma[0], n_density_points)
    y = np.linspace(-xy_lim_sigma * bunch1.sigma[1], xy_lim_sigma * bunch1.sigma[1], n_density_points)
    z = np.linspace(-z_lim_sigma * bunch1.sigma[2], z_lim_sigma * bunch1.sigma[2], n_density_points + 5)

    z_cm = z / 1e4

    X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z)  # For 3D space

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

    # # Set up the figure for density product visualization
    # fig, ax = plt.subplots(2, 1)
    # im_xz = ax[0].imshow(np.zeros((x.size, z.size)), extent=[z.min(), z.max(), x.min(), x.max()],
    #                      origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_product, aspect='auto')
    # im_yz = ax[1].imshow(np.zeros((y.size, z.size)), extent=[z.min(), z.max(), y.min(), y.max()],
    #                      origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_product, aspect='auto')
    #
    # ax[0].set_title('Density Product in x-z and y-x Planes')
    # ax[0].set_xlabel('z (um)')
    # ax[0].set_ylabel('x (um)')
    # ax[1].set_xlabel('z (um)')
    # ax[1].set_ylabel('y (um)')
    # # Set vertical space between axes to zero
    # fig.subplots_adjust(hspace=0)
    #
    # # Set up a separate figure for density sum visualization
    # fig_sum, ax_sum = plt.subplots(2, 1)
    # im_sum_xz = ax_sum[0].imshow(np.zeros((x.size, z.size)), extent=[z.min(), z.max(), x.min(), x.max()],
    #                              origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_sum, aspect='auto')
    # im_sum_yz = ax_sum[1].imshow(np.zeros((y.size, z.size)), extent=[z.min(), z.max(), y.min(), y.max()],
    #                              origin='lower', cmap='jet', vmin=0, vmax=max_bunch_density_sum, aspect='auto')
    #
    # ax_sum[0].set_title('Density Sum in x-z and y-z Planes')
    # ax_sum[0].set_xlabel('z (um)')
    # ax_sum[0].set_ylabel('x (um)')
    # ax_sum[1].set_xlabel('z (um)')
    # ax_sum[1].set_ylabel('y (um)')
    # fig_sum.subplots_adjust(hspace=0)

    integrated_density_product_z = np.zeros_like(z)
    fig_z_slices, ax_z_slices = plt.subplots(2, 1)

    first_pass = True
    im_xzs, im_yzs, im_sum_xzs, im_sum_yzs = [], [], [], []
    for i in range(n_propagation_points):
        density1_xyz = bunch1.density(X_3d, Y_3d, Z_3d)
        density2_xyz = bunch2.density(X_3d, Y_3d, Z_3d)

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
    bunch.set_r(0., 0., 0.)  # um Initial position of bunch
    bunch.set_beta(0., 0., 1.)  # Dimensionless velocity of bunch
    bunch.set_sigma(10., 10., 100.)  # um Width of bunch in x, y, z
    bunch.dt = 1e-5  # ns Timestep to propagate bunch

    n_propagation_points = 100
    n_density_points = 100

    # Create a grid of points for the x-z and y-z planes
    x = np.linspace(-2 * bunch.sigma[0], 2 * bunch.sigma[0], n_density_points)
    y = np.linspace(-2 * bunch.sigma[1], 2 * bunch.sigma[1], n_density_points)
    z = np.linspace(-2 * bunch.sigma[2], 2 * bunch.sigma[2], n_density_points)

    X, Z = np.meshgrid(x, z)  # For x-z plane
    Y, Z_yz = np.meshgrid(y, z)  # For y-z plane

    fig, ax = plt.subplots(2, 1)
    im_xz = ax[0].imshow(np.zeros((n_density_points, n_density_points)), extent=[z.min(), z.max(), x.min(), x.max()],
                         origin='lower', cmap='jet', vmin=0, vmax=0.8)
    im_yz = ax[1].imshow(np.zeros((n_density_points, n_density_points)), extent=[z.min(), z.max(), y.min(), y.max()],
                         origin='lower', cmap='jet', vmin=0, vmax=0.8)

    ax[0].set_title('Bunch Propagation in x-z Plane')
    ax[1].set_title('Bunch Propagation in y-z Plane')
    ax[0].set_xlabel('z (um)')
    ax[0].set_ylabel('x (um)')
    ax[1].set_xlabel('z (um)')
    ax[1].set_ylabel('y (um)')
    print(Z)
    print(X)

    for i in range(n_propagation_points):
        # Calculate the density in the x-z and y-z planes
        density_xz = bunch.density(X, 0., Z)
        density_yz = bunch.density(0., Y, Z_yz)

        im_xz.set_data(density_xz.transpose())
        im_yz.set_data(density_yz.transpose())

        plt.pause(0.01)
        bunch.propagate()

    plt.show()


def gaus_1d(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def vernier_scan_fit(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))


class BunchDensity:
    """
    Class to hold particle density for a single bunch. Currently use a 3D gaussian distribution in lab reference frame
    (Lorentz contracted) to represent the bunch density. Use the center of the bunch as the origin of the distribution.
    Bunch also has a 3D velocity vector and a method to calculate the density at a given point in the lab frame.
    """
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light

    def __init__(self):
        self.sigma = np.array([0., 0., 0.], dtype=np.float64)  # um Width of gaussian bunch profile in x, y, z
        self.beta = np.array([0., 0., 0.], dtype=np.float64)  # v/c Dimensionless velocity of bunch in x, y, z
        self.r = np.array([0., 0., 0.], dtype=np.float64)  # um Position of bunch center in x, y, z
        self.t = 0.  # ns Time of bunch motion
        self.dt = 0.  # ns Timestep to propagate bunch
        self.angle = 0.  # Rotation angle in radians
        self.beta_star = None  # cm Beta star value for the bunch

    def set_r(self, x, y, z):
        """
        Set the position of the bunch.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        """
        self.r = np.array([x, y, z], dtype=np.float64)

    def set_beta(self, x, y, z):
        """
        Set the velocity of the bunch.
        :param x: float x velocity in lab frame
        :param y: float y velocity in lab frame
        :param z: float z velocity in lab frame
        """
        self.beta = np.array([x, y, z], dtype=np.float64)

    def set_sigma(self, x, y, z):
        """
        Set the width of the bunch in the lab frame.
        :param x: float x width in lab frame
        :param y: float y width in lab frame
        :param z: float z width in lab frame
        """
        self.sigma = np.array([x, y, z], dtype=np.float64)

    def set_angle(self, angle):
        """
        Set the rotation angle of the bunch in the y-z plane and adjust r and beta accordingly.
        :param angle: float Rotation angle in radians
        """
        self.angle = angle

        # Define rotation matrix for y-z plane
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rotate position vector r
        y_rot = self.r[1] * cos_angle + self.r[2] * sin_angle
        z_rot = -self.r[1] * sin_angle + self.r[2] * cos_angle
        self.r[1] = y_rot
        self.r[2] = z_rot

        # Rotate velocity vector beta
        beta_y_rot = self.beta[1] * cos_angle + self.beta[2] * sin_angle
        beta_z_rot = -self.beta[1] * sin_angle + self.beta[2] * cos_angle
        self.beta[1] = beta_y_rot
        self.beta[2] = beta_z_rot

    def density(self, x, y, z):
        """
        Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        :return: float Density of bunch at given point
        """
        return bdcpp.density(x, y, z, self.r[0], self.r[1], self.r[2],
                             self.sigma[0], self.sigma[1], self.sigma[2],
                             self.angle, self.beta_star if self.beta_star is not None else 0)

    def density_py(self, x, y, z):
        """
        Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        :return: float Density of bunch at given point
        """
        # Broadening along the z-axis
        z_rel = z - self.r[2]
        if self.beta_star is None:
            sigma_x = self.sigma[0]
            sigma_y = self.sigma[1]
        else:
            sigma_x = self.sigma[0] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
            sigma_y = self.sigma[1] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
            # sigma_x = self.sigma[0] * np.sqrt(1 + abs(z) / 1e5)
            # sigma_y = self.sigma[1] * np.sqrt(1 + abs(z) / 1e5)

        # Rotate coordinates in the y-z plane
        y_rot = (y - self.r[1]) * np.cos(self.angle) - z_rel * np.sin(self.angle)
        z_rot = (y - self.r[1]) * np.sin(self.angle) + z_rel * np.cos(self.angle)

        # Calculate the density using the modified sigma_x, sigma_y, and rotated coordinates
        x_rel = x - self.r[0]
        density = np.exp(
            -0.5 * (x_rel ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2 + z_rot ** 2 / self.sigma[2] ** 2))
        density /= (2 * np.pi) ** 1.5 * sigma_x * sigma_y * self.sigma[2]  # Normalize the exponential

        return density

    def propagate(self):
        """
        Propagate the bunch to the next timestep.
        """
        self.r += self.beta * self.c * self.dt
        self.t += self.dt


class BunchCollider:
    def __init__(self):
        self.bunch1 = BunchDensity()
        self.bunch2 = BunchDensity()

        self.bunch1_beta_original = np.array([0., 0., +1.])
        self.bunch2_beta_original = np.array([0., 0., -1.])
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

        self.bunch1.set_sigma(150., 150., 1.1e6)
        self.bunch2.set_sigma(150., 150., 1.1e6)  # microns

        self.bunch1.beta_star = 85  # cm
        self.bunch2.beta_star = 85  # cm

        self.bunch1_r_original = np.array([0., 0., -6.e6])
        self.bunch2_r_original = np.array([0., 0., +6.e6])
        self.bunch1.set_r(*self.bunch1_r_original)
        self.bunch2.set_r(*self.bunch2_r_original)

        self.x_lim_sigma = 10
        self.y_lim_sigma = 10
        self.z_lim_sigma = 7

        self.n_points_x = 101
        self.n_points_y = 101
        self.n_points_z = 101
        self.n_points_t = 50

        self.x, self.y, self.z = None, None, None
        self.average_density_product_xyz = None

    def set_bunch_sigmas(self, sigma1, sigma2):
        self.bunch1.set_sigma(*sigma1)
        self.bunch2.set_sigma(*sigma2)

    def set_bunch_betas(self, beta1, beta2):
        self.bunch1_beta_original = beta1
        self.bunch2_beta_original = beta2
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

    def set_bunch_rs(self, r1, r2):
        self.bunch1_r_original = r1
        self.bunch2_r_original = r2
        self.bunch1.set_r(*self.bunch1_r_original)
        self.bunch2.set_r(*self.bunch2_r_original)

    def set_bunch_beta_stars(self, beta_star1, beta_star2):
        self.bunch1.beta_star = beta_star1
        self.bunch2.beta_star = beta_star2

    def set_bunch_crossing(self, crossing_angle1, crossing_angle2):
        self.bunch1.set_angle(crossing_angle1)
        self.bunch2.set_angle(crossing_angle2)

    def run_sim(self):
        # Reset
        self.bunch1.set_r(*self.bunch1_r_original)
        self.bunch2.set_r(*self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angle(self.bunch1.angle)
        self.bunch2.set_angle(self.bunch2.angle)
        self.average_density_product_xyz = None

        # Set timestep for propagation
        dt = (self.bunch2.r[2] - self.bunch1.r[2]) / self.bunch1.c / self.n_points_t
        self.bunch1.dt = self.bunch2.dt = dt  # ns Timestep to propagate both bunches

        # Create a grid of points for the x-z and y-z planes
        self.x = np.linspace(-self.x_lim_sigma * self.bunch1.sigma[0], self.x_lim_sigma * self.bunch1.sigma[0],
                             self.n_points_x)
        self.y = np.linspace(-self.y_lim_sigma * self.bunch1.sigma[1], self.y_lim_sigma * self.bunch1.sigma[1],
                             self.n_points_y)
        self.z = np.linspace(-self.z_lim_sigma * self.bunch1.sigma[2], self.z_lim_sigma * self.bunch1.sigma[2],
                             self.n_points_z)

        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z)  # For 3D space

        for i in range(self.n_points_t):
            density1_xyz = self.bunch1.density(x_3d, y_3d, z_3d)
            density2_xyz = self.bunch2.density(x_3d, y_3d, z_3d)

            # Calculate the density product
            density_product_xyz = density1_xyz * density2_xyz
            if self.average_density_product_xyz is None:
                self.average_density_product_xyz = density_product_xyz
            else:
                self.average_density_product_xyz += density_product_xyz

            self.bunch1.propagate()
            self.bunch2.propagate()
        self.average_density_product_xyz /= self.n_points_t

    def compute_time_step(self, time_step_index):
        """
        Compute the density for a specific time step.
        :param time_step_index: Index of the time step to compute
        :return: Density product for the given time step
        """
        self.bunch1.propagate()
        self.bunch2.propagate()

        # Create a grid of points for the x-y-z planes
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z)

        density1_xyz = self.bunch1.density(x_3d, y_3d, z_3d)
        density2_xyz = self.bunch2.density(x_3d, y_3d, z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        return density_product_xyz

    def run_sim_parallel(self):
        # Reset
        self.bunch1.set_r(*self.bunch1_r_original)
        self.bunch2.set_r(*self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angle(self.bunch1.angle)
        self.bunch2.set_angle(self.bunch2.angle)
        self.average_density_product_xyz = None

        # Set timestep for propagation
        dt = (self.bunch2.r[2] - self.bunch1.r[2]) / self.bunch1.c / self.n_points_t
        self.bunch1.dt = self.bunch2.dt = dt

        # Create a grid of points for the x-z and y-z planes
        self.x = np.linspace(-self.x_lim_sigma * self.bunch1.sigma[0], self.x_lim_sigma * self.bunch1.sigma[0],
                             self.n_points_x)
        self.y = np.linspace(-self.y_lim_sigma * self.bunch1.sigma[1], self.y_lim_sigma * self.bunch1.sigma[1],
                             self.n_points_y)
        self.z = np.linspace(-self.z_lim_sigma * self.bunch1.sigma[2], self.z_lim_sigma * self.bunch1.sigma[2],
                             self.n_points_z)

        # Create a pool of workers to compute each time step in parallel
        with Pool() as pool:
            results = pool.map(self.compute_time_step, range(self.n_points_t))

        # Accumulate the results
        self.average_density_product_xyz = np.sum(results, axis=0) / self.n_points_t

    def get_beam_sigmas(self):
        return self.bunch1.sigma, self.bunch2.sigma

    def get_z_density_dist(self):
        return self.z, np.sum(self.average_density_product_xyz, axis=(0, 1))

    def get_param_string(self):
        print(f'Beam1.r: {self.bunch1.r}')
        print(f'Beam1_r_original: {self.bunch1_r_original}')
        print(f'{self.bunch1.r}')
        print(f'{self.bunch1.r[:2]}')
        param_string = (f'Beta*s = {self.bunch1.beta_star:.1f}, {self.bunch2.beta_star:.1f} cm\n'
                        f'Beam Widths = {self.bunch1.sigma[0]:.1f},{self.bunch2.sigma[0]:.1f} um\n'
                        f'Beam Lengths = {self.bunch1.sigma[2] / 1e4:.1f}, {self.bunch2.sigma[2] / 1e4:.1f} cm\n'
                        f'Crossing Angles = {self.bunch1.angle * 1e3:.1f}, {self.bunch2.angle * 1e3:.1f} mrad\n'
                        f'Beam Offsets = {np.sqrt(np.sum(self.bunch1.r[:2] ** 2)):.1f}, '
                        f'{np.sqrt(np.sum(self.bunch2.r[:2] ** 2)):.1f} um')
        return param_string


if __name__ == '__main__':
    main()
