#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 30 2:32 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/BunchCollider.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from multiprocessing import Pool

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

        print(f'Bunch1.r: {self.bunch1.r}')
        print(f'Bunch2.r: {self.bunch2.r}')
        print(f'Bunch1_r_original: {self.bunch1_r_original}')
        print(f'Bunch2_r_original: {self.bunch2_r_original}')
        print(f'Bunch1.beta: {self.bunch1.beta}')
        print(f'Bunch2.beta: {self.bunch2.beta}')
        print(f'Bunch1_beta_original: {self.bunch1_beta_original}')
        print(f'Bunch2_beta_original: {self.bunch2_beta_original}')
        print(f'Bunch1.angle: {self.bunch1.angle}')
        print(f'Bunch2.angle: {self.bunch2.angle}')
        print(f'Bunch1.sigma: {self.bunch1.sigma}')
        print(f'Bunch2.sigma: {self.bunch2.sigma}')
        print(f'Bunch1.beta_star: {self.bunch1.beta_star}')
        print(f'Bunch2.beta_star: {self.bunch2.beta_star}')

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

        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # For 3D space

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
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')

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
                        f'Crossing Angles = {self.bunch1.angle * 1e3:.2f}, {self.bunch2.angle * 1e3:.2f} mrad\n'
                        f'Beam Offsets = {np.sqrt(np.sum(self.bunch1_r_original[:2] ** 2)):.0f}, '
                        f'{np.sqrt(np.sum(self.bunch2_r_original[:2] ** 2)):.0f} um')
        return param_string
