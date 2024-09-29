#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 30 2:33 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/BunchDensity.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from scipy.optimize import curve_fit as cf
import copy
import bunch_density_cpp as bdcpp


class BunchDensity:
    """
    Class to hold particle density for a single bunch. Currently use a 3D gaussian distribution in lab reference frame
    (Lorentz contracted) to represent the bunch density. Use the center of the bunch as the origin of the distribution.
    Bunch also has a 3D velocity vector and a method to calculate the density at a given point in the lab frame.
    """
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light

    def __init__(self):
        self.transverse_sigma = np.array([0., 0.], dtype=np.float64)  # um Width of gaussian bunch profile in x, y
        self.beta = np.array([0., 0., 1.], dtype=np.float64)  # v/c Dimensionless velocity of bunch in x, y, z
        self.r = np.array([0., 0., 0.], dtype=np.float64)  # um Position of bunch center in x, y, z
        self.t = 0.  # ns Time of bunch motion
        self.dt = 0.  # ns Timestep to propagate bunch
        self.angle_x = 0.  # Rotation angle in y-z plane in radians
        self.angle_y = 0.  # Rotation angle in x-z plane in radians
        self.beta_star = None  # cm Beta star value for the bunch
        self.delay = 0.  # ns Time delay of the bunch

        self.longitudinal_params = {'mu1': 0., 'sigma1': 1., 'a2': 0., 'mu2': 0., 'sigma2': 1.,
                                    'a3': 0., 'mu3': 0., 'sigma3': 1., 'a4': 0., 'mu4': 0., 'sigma4': 1.}
        self.longitudinal_width_scaling = 1.  # Scaling factor for the longitudinal beam profile

        self.initial_z = 0.  # Initial z distance in um
        self.offset_x = 0.  # Offset in x in um
        self.offset_y = 0.  # Offset in y in um

        self.reset = True  # If true calculate r and bete before next density calculation, if false use current values

    def set_initial_z(self, z):
        """
        Set the initial z distance of the bunch.
        :param z: float Initial z position in lab frame
        """
        self.initial_z = z
        self.reset = True

    def set_offsets(self, x_offset, y_offset):
        """
        Set the x and y offsets for the bunch.
        :param x_offset: float x offset in lab frame
        :param y_offset: float y offset in lab frame
        """
        self.offset_x = x_offset
        self.offset_y = y_offset
        self.reset = True

    def set_beta(self, x, y, z):
        """
        Set the velocity of the bunch.
        :param x: float x velocity in lab frame
        :param y: float y velocity in lab frame
        :param z: float z velocity in lab frame
        """
        self.beta = np.array([x, y, z], dtype=np.float64)
        self.reset = True

    def set_beta_star(self, beta_star):
        """
        Set the beta star value for the bunch.
        :param beta_star: float Beta star value in cm
        """
        self.beta_star = beta_star
        self.reset = True

    def set_sigma(self, x, y, z=None):
        """
        Set the width of the bunch in the lab frame.
        :param x: float x width in lab frame
        :param y: float y width in lab frame
        :param z: float z width in lab frame
        """
        self.transverse_sigma = np.array([x, y], dtype=np.float64)
        if z is not None:
            self.longitudinal_params['sigma1'] = z
        self.reset = True

    def set_angles(self, angle_x, angle_y):
        """
        Set the rotation angles of the bunch in the y-z and x-z planes.
        :param angle_x: float Rotation angle in y-z plane in radians
        :param angle_y: float Rotation angle in x-z plane in radians
        """
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.reset = True

    def set_delay(self, delay):
        """
        Set the time delay of the bunch.
        :param delay: float Time delay in ns
        """
        self.delay = delay
        self.reset = True

    def read_longitudinal_beam_profile_fit_parameters_from_file(self, fit_out_path):
        """
        Read the longitudinal beam profile fit parameters from a file and set them.
        :param fit_out_path: str Path to file with fit parameters
        """
        fit_params = read_longitudinal_beam_profile_fit_parameters(fit_out_path)
        self.longitudinal_params = fit_params
        self.reset = True

    def set_longitudinal_beam_profile_scaling(self, scaling):
        """
        Set the scaling of the longitudinal beam profile. Scale all fit sigmas by this factor.
        :param scaling: float Scaling factor for the longitudinal beam profile
        """
        self.longitudinal_width_scaling = scaling
        self.longitudinal_params['sigma1'] *= scaling
        self.longitudinal_params['sigma2'] *= scaling
        self.longitudinal_params['sigma3'] *= scaling
        self.longitudinal_params['sigma4'] *= scaling
        self.reset = True

    def get_beam_length(self):
        """
        Calculate the length of the bunch based on a single gaus fit of the quad_gaus function with
        the longitudinal fit parameters.
        """
        mu1 = self.longitudinal_params['mu1']
        sigma1 = self.longitudinal_params['sigma1']
        a2 = self.longitudinal_params['a2']
        mu2 = self.longitudinal_params['mu2']
        sigma2 = self.longitudinal_params['sigma2']
        a3 = self.longitudinal_params['a3']
        mu3 = self.longitudinal_params['mu3']
        sigma3 = self.longitudinal_params['sigma3']
        a4 = self.longitudinal_params['a4']
        mu4 = self.longitudinal_params['mu4']
        sigma4 = self.longitudinal_params['sigma4']

        x = np.linspace(-abs(self.initial_z), abs(self.initial_z), 1000)
        y = quad_gaus_pdf(x, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4)
        p0 = [mu1, sigma1]
        popt, _ = cf(gaus_pdf, x, y, p0)
        length = 2 * popt[1]

        return length

    def calculate_r_and_beta(self):
        """
        Calculate the r position and beta based on initial z distance, x, y offsets, and angles.
        The total distance from the origin to the center of the bunch (ignoring offsets) should be the initial z distance.
        After the center is calculated, apply the x and y offsets.
        """
        # Calculate the delayed z position based on the initial z distance and delay. Positive delay, larger z
        delayed_z = self.initial_z + np.sign(self.initial_z) * self.delay * self.c

        # Calculate the unrotated position vector based on the initial z distance
        r_rotated = delayed_z * np.array([np.sin(self.angle_x), np.sin(self.angle_y), 1.], dtype=np.float64)

        # Apply x and y offsets (only to r, not beta)
        r_final = r_rotated + np.array([self.offset_x, self.offset_y, 0.], dtype=np.float64)

        self.r = r_final

        # Calculate beta based on the rotated position (ignoring offsets)
        beta_direction = -r_rotated / np.linalg.norm(r_rotated)
        self.beta = beta_direction

        self.t = 0
        self.reset = False

    def density(self, x, y, z):
        """
        Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        :return: float Density of bunch at given point
        """
        if self.reset:
            self.calculate_r_and_beta()
        return bdcpp.density(x, y, z, self.r[0], self.r[1], self.r[2],
                             self.transverse_sigma[0], self.transverse_sigma[1],
                             self.angle_x, self.angle_y, self.beta_star if self.beta_star is not None else 0,
                             *self.longitudinal_params.values())

    def density_py(self, x, y, z):
        """
        Calculate the density of the bunch at a given point in the lab frame, considering the y and x angles.
        The density is calculated relative to the bunch's direction, rather than the coordinate system.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        :return: float Density of bunch at given point
        """
        if self.reset:
            self.calculate_r_and_beta()
            self.reset = False

        # Calculate the relative position vector
        relative_r = np.array([x - self.r[0], y - self.r[1], z - self.r[2]], dtype=np.float64)

        # Apply rotation for the xz plane (rotation around the y-axis)
        cos_xz = np.cos(self.angle_x)
        sin_xz = np.sin(self.angle_x)
        x_rot = relative_r[0] * cos_xz - relative_r[2] * sin_xz
        z_rot_xz = relative_r[0] * sin_xz + relative_r[2] * cos_xz

        # Update relative_r with the rotation result in the xz plane
        relative_r[0] = x_rot
        relative_r[2] = z_rot_xz

        # Apply rotation for the yz plane (rotation around the x-axis)
        cos_yz = np.cos(self.angle_y)
        sin_yz = np.sin(self.angle_y)
        y_rot = relative_r[1] * cos_yz - relative_r[2] * sin_yz
        z_rot_yz = relative_r[1] * sin_yz + relative_r[2] * cos_yz

        # Update relative_r with the rotation result in the yz plane
        relative_r[1] = y_rot
        relative_r[2] = z_rot_yz

        # Broadening along the z-axis (after rotation)
        if self.beta_star is None:
            sigma_x = self.transverse_sigma[0]
            sigma_y = self.transverse_sigma[1]
        else:
            # Calculate distance to IP along the axis of travel, based on z and the angles
            distance_to_IP = z * np.sqrt(1 + np.tan(self.angle_x) ** 2 + np.tan(self.angle_y) ** 2)
            sigma_x = self.transverse_sigma[0] * np.sqrt(1 + distance_to_IP ** 2 / (self.beta_star * 1e4) ** 2)
            sigma_y = self.transverse_sigma[1] * np.sqrt(1 + distance_to_IP ** 2 / (self.beta_star * 1e4) ** 2)

        # Calculate the density using the modified sigma_x, sigma_y, and rotated coordinates
        density = np.exp(
            -0.5 * (relative_r[0] ** 2 / sigma_x ** 2 +
                    relative_r[1] ** 2 / sigma_y ** 2 +
                    relative_r[2] ** 2 / self.transverse_sigma[2] ** 2)
        )
        density /= (2 * np.pi) ** 1.5 * sigma_x * sigma_y * self.transverse_sigma[2]  # Normalize the exponential

        return density

    def propagate(self):
        """
        Propagate the bunch to the next timestep.
        """
        self.r += self.beta * self.c * self.dt
        self.t += self.dt

    def propagate_n_steps(self, n):
        """
        Propagate the bunch n timesteps.
        :param n: int Number of timesteps to propagate
        """
        self.r += self.beta * self.c * self.dt * n
        self.t += self.dt * n

    def copy(self):
        """
        Return a deep copy of the bunch.
        :return: BunchDensity Deep copy of the bunch
        """
        return copy.deepcopy(self)

    def __str__(self):
        return (f'Bunch Parameters:\n'
                f'Initial Z: {self.initial_z:.0f} um\n'
                f'Offsets: {self.offset_x:.4f}, {self.offset_y:.4f} um\n'
                f'Beta: {self.beta}\n'
                f'Sigma: {self.transverse_sigma}\n'
                f'Angles: {self.angle_x * 1e3:.4f}, {self.angle_y * 1e3:.4f} mrads\n'
                f'Position: {self.r}\n'
                f'Time: {self.t} ns\n'
                f'Timestep: {self.dt} ns\n'
                f'Beta Star: {self.beta_star:.1f} cm')


def read_longitudinal_beam_profile_fit_parameters(fit_out_path):
    with open(fit_out_path, 'r') as file:
        lines = file.readlines()

    fit_parameters = {}
    for line in lines[3:]:
        param, val = line.strip().split(': ')
        fit_parameters[param] = float(val)

    return fit_parameters


def gaus_pdf(x, b, c):
    return np.exp(-(x - b)**2 / (2 * c**2)) / (c * np.sqrt(2 * np.pi))


def quad_gaus_pdf(x, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return (gaus_pdf(x, b1, c1) + a2 * gaus_pdf(x, b2, c2) + a3 * gaus_pdf(x, b3, c3) + a4 * gaus_pdf(x, b4, c4)) / (
            1 + a2 + a3 + a4)
