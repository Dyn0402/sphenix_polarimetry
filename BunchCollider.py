#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 30 2:32 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/BunchCollider.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool
from scipy.ndimage import gaussian_filter1d

from BunchDensity import BunchDensity


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
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)

        self.z_shift = 0.  # microns Distance to shift the center of the collisions along beam axis
        self.amplitude = 1.  # arb Scale amplitude of z-distribution by this amount

        self.bkg = 1e-20 * 0.  # Background density to add to the density product, interpret as air density

        self.z_bounds = (-265. * 1e4, 265. * 1e4)

        self.gaus_smearing_sigma = None
        self.gaus_z_efficiency_width = None

        self.x_lim_sigma = 10
        self.y_lim_sigma = 10
        self.z_lim_sigma = 5

        self.n_points_x = 61
        self.n_points_y = 61
        self.n_points_z = 151
        self.n_points_t = 60

        self.bunch1_longitudinal_fit_parameter_path = None
        self.bunch2_longitudinal_fit_parameter_path = None
        self.bunch1_longitudinal_fit_scaling = 1.
        self.bunch2_longitudinal_fit_scaling = 1.

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
        self.bunch1.set_initial_z(self.bunch1_r_original[2])
        self.bunch2.set_initial_z(self.bunch2_r_original[2])
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_offsets(self, offset1, offset2):
        self.bunch1_r_original[:2] = offset1
        self.bunch2_r_original[:2] = offset2
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_beta_stars(self, beta_star1, beta_star2):
        self.bunch1.beta_star = beta_star1
        self.bunch2.beta_star = beta_star2

    def set_bunch_crossing(self, crossing_angle1_x, crossing_angle1_y, crossing_angle2_x, crossing_angle2_y):
        self.bunch1.set_angles(crossing_angle1_x, crossing_angle1_y)
        self.bunch2.set_angles(crossing_angle2_x, crossing_angle2_y)

    def set_z_shift(self, z_shift):
        self.z_shift = z_shift

    def set_amplitude(self, amp):
        self.amplitude = amp

    def set_z_bounds(self, z_bounds):
        self.z_bounds = z_bounds

    def set_gaus_smearing_sigma(self, sigma):
        self.gaus_smearing_sigma = sigma

    def set_gaus_z_efficiency_width(self, width):
        self.gaus_z_efficiency_width = width

    def set_bunch_delays(self, delay1, delay2):
        self.bunch1.set_delay(delay1)
        self.bunch2.set_delay(delay2)

    def set_bkg(self, bkg):
        self.bkg = bkg

    def set_longitudinal_fit_parameters_from_file(self, bunch1_path, bunch2_path):
        self.bunch1_longitudinal_fit_parameter_path = bunch1_path
        self.bunch2_longitudinal_fit_parameter_path = bunch2_path
        self.bunch1.read_longitudinal_beam_profile_fit_parameters_from_file(bunch1_path)
        self.bunch2.read_longitudinal_beam_profile_fit_parameters_from_file(bunch2_path)

    def set_longitudinal_fit_scaling(self, scale1, scale2):
        self.bunch1_longitudinal_fit_scaling = scale1
        self.bunch2_longitudinal_fit_scaling = scale2
        self.bunch1.set_longitudinal_beam_profile_scaling(scale1)
        self.bunch2.set_longitudinal_beam_profile_scaling(scale2)

    def run_sim(self, print_params=False):
        # Reset
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz = None

        # Set timestep for propagation
        dt = (self.bunch2.initial_z - self.bunch1.initial_z) / self.bunch1.c / self.n_points_t
        self.bunch1.dt = self.bunch2.dt = dt  # ns Timestep to propagate both bunches

        # Create a grid of points for the x-z and y-z planes
        self.x = np.linspace(-self.x_lim_sigma * self.bunch1.transverse_sigma[0], self.x_lim_sigma * self.bunch1.transverse_sigma[0],
                             self.n_points_x)
        self.y = np.linspace(-self.y_lim_sigma * self.bunch1.transverse_sigma[1], self.y_lim_sigma * self.bunch1.transverse_sigma[1],
                             self.n_points_y)
        if self.z_bounds is not None:
            self.z = np.linspace(self.z_bounds[0], self.z_bounds[1], self.n_points_z)
        else:
            min_z = min(self.bunch1_r_original[2], self.bunch2_r_original[2])
            max_z = max(self.bunch1_r_original[2], self.bunch2_r_original[2])
            self.z = np.linspace(min_z - self.z_lim_sigma * self.bunch1.transverse_sigma[2],
                                 max_z + self.z_lim_sigma * self.bunch1.transverse_sigma[2], self.n_points_z)

        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # For 3D space
        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        for i in range(self.n_points_t):
            density1_xyz = self.bunch1.density(x_3d, y_3d, z_3d)
            density2_xyz = self.bunch2.density(x_3d, y_3d, z_3d)

            # Calculate the density product
            density_product_xyz = density1_xyz * density2_xyz

            # Add background
            density_product_xyz += density1_xyz * self.bkg + density2_xyz * self.bkg

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
        bunch1_copy = self.bunch1.copy()
        bunch2_copy = self.bunch2.copy()
        bunch1_copy.propagate_n_steps(time_step_index)
        bunch2_copy.propagate_n_steps(time_step_index)

        # Create a grid of points for the x-y-z planes
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        density1_xyz =  bunch1_copy.density(x_3d, y_3d, z_3d)
        density2_xyz = bunch2_copy.density(x_3d, y_3d, z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        # Add background
        density_product_xyz += density1_xyz * self.bkg + density2_xyz * self.bkg

        return density_product_xyz

    def run_sim_parallel(self, print_params=False):
        # Reset
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz = None

        # Set timestep for propagation
        dt = (self.bunch2.initial_z - self.bunch1.initial_z) / self.bunch1.c / self.n_points_t
        self.bunch1.dt = self.bunch2.dt = dt  # ns Timestep to propagate both bunches

        # Create a grid of points for the x-z and y-z planes
        self.x = np.linspace(-self.x_lim_sigma * self.bunch1.transverse_sigma[0],
                             self.x_lim_sigma * self.bunch1.transverse_sigma[0],
                             self.n_points_x)
        self.y = np.linspace(-self.y_lim_sigma * self.bunch1.transverse_sigma[1],
                             self.y_lim_sigma * self.bunch1.transverse_sigma[1],
                             self.n_points_y)
        if self.z_bounds is not None:
            self.z = np.linspace(self.z_bounds[0], self.z_bounds[1], self.n_points_z)
        else:
            min_z = min(self.bunch1_r_original[2], self.bunch2_r_original[2])
            max_z = max(self.bunch1_r_original[2], self.bunch2_r_original[2])
            self.z = np.linspace(min_z - self.z_lim_sigma * self.bunch1.transverse_sigma[2],
                                 max_z + self.z_lim_sigma * self.bunch1.transverse_sigma[2], self.n_points_z)

        # x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # For 3D space
        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        self.average_density_product_xyz = 0
        with Pool() as pool:
            density_products = pool.map(self.compute_time_step, range(self.n_points_t))
        self.average_density_product_xyz = np.mean([p for p in density_products], axis=0)

    def get_beam_sigmas(self):
        return self.bunch1.transverse_sigma, self.bunch2.transverse_sigma

    def get_z_density_dist(self):
        z_vals = (self.z - self.z_shift) / 1e4  # um to cm
        z_dist = self.amplitude * np.sum(self.average_density_product_xyz, axis=(0, 1))
        if self.gaus_z_efficiency_width is not None:
            z_dist = z_dist * gaus(z_vals, 1, 0, self.gaus_z_efficiency_width)
        if self.gaus_smearing_sigma is not None:
            z_spacing = z_vals[1] - z_vals[0]
            z_dist = gaussian_filter1d(z_dist, self.gaus_smearing_sigma / z_spacing)
        return z_vals, z_dist

    def get_param_string(self):
        param_string = (f'Beta*s: {self.bunch1.beta_star:.1f}, {self.bunch2.beta_star:.1f} cm\n'
                        f'Beam Widths: {self.bunch1.transverse_sigma[0]:.1f}, {self.bunch2.transverse_sigma[0]:.1f} um\n'
                        f'Beam Lengths: {self.bunch1.get_beam_length() / 1e4:.1f}, {self.bunch2.get_beam_length() / 1e4:.1f} cm\n'
                        f'Crossing Angles y: {self.bunch1.angle_y * 1e3:.2f}, {self.bunch2.angle_y * 1e3:.2f} mrad\n'
                        f'Crossing Angles x: {self.bunch1.angle_x * 1e3:.2f}, {self.bunch2.angle_x * 1e3:.2f} mrad\n'
                        f'Beam Offsets: {np.sqrt(np.sum(self.bunch1_r_original[:2] ** 2)):.0f}, '
                        f'{np.sqrt(np.sum(self.bunch2_r_original[:2] ** 2)):.0f} um')
        return param_string

    def __str__(self):
        return (f'BunchCollider:\n'
                f'z_shift: {self.z_shift}, amplitude: {self.amplitude}, x_lim_sigma: {self.x_lim_sigma}, '
                f'y_lim_sigma: {self.y_lim_sigma}, z_lim_sigma: {self.z_lim_sigma}, n_points_x: {self.n_points_x}, '
                f'n_points_y: {self.n_points_y}, n_points_z: {self.n_points_z}, n_points_t: {self.n_points_t}, '
                f'bunch1_r_original: {self.bunch1_r_original}, bunch2_r_original: {self.bunch2_r_original}, '
                f'bunch1_beta_original: {self.bunch1_beta_original}, bunch2_beta_original: {self.bunch2_beta_original}\n'
                f'\nbunch1: {self.bunch1}\n'
                f'\nbunch2: {self.bunch2}\n')


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
