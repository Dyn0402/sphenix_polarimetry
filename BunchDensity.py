#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 30 2:33 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/BunchDensity.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import bunch_density_cpp as bdcpp


class BunchDensity:
    """
    Class to hold particle density for a single bunch. Currently use a 3D gaussian distribution in lab reference frame
    (Lorentz contracted) to represent the bunch density. Use the center of the bunch as the origin of the distribution.
    Bunch also has a 3D velocity vector and a method to calculate the density at a given point in the lab frame.
    """
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light

    def __init__(self):
        self.sigma = np.array([0., 0., 0.], dtype=np.float64)  # um Width of gaussian bunch profile in x, y, z
        self.beta = np.array([0., 0., 1.], dtype=np.float64)  # v/c Dimensionless velocity of bunch in x, y, z
        self.r = np.array([0., 0., 0.], dtype=np.float64)  # um Position of bunch center in x, y, z
        self.t = 0.  # ns Time of bunch motion
        self.dt = 0.  # ns Timestep to propagate bunch
        self.angle_x = 0.  # Rotation angle in y-z plane in radians
        self.angle_y = 0.  # Rotation angle in x-z plane in radians
        self.beta_star = None  # cm Beta star value for the bunch

        self.initial_z = 0.  # Initial z distance in um
        self.offset_x = 0.  # Offset in x in um
        self.offset_y = 0.  # Offset in y in um

    def set_initial_z(self, z):
        """
        Set the initial z distance of the bunch.
        :param z: float Initial z position in lab frame
        """
        self.initial_z = z

    def set_offsets(self, x_offset, y_offset):
        """
        Set the x and y offsets for the bunch.
        :param x_offset: float x offset in lab frame
        :param y_offset: float y offset in lab frame
        """
        self.offset_x = x_offset
        self.offset_y = y_offset

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

    def set_angles(self, angle_x, angle_y):
        """
        Set the rotation angles of the bunch in the y-z and x-z planes.
        :param angle_x: float Rotation angle in y-z plane in radians
        :param angle_y: float Rotation angle in x-z plane in radians
        """
        self.angle_x = angle_x
        self.angle_y = angle_y

    def calculate_r_and_beta(self):
        """
        Calculate the r position and beta based on initial z distance, x, y offsets, and angles.
        The total distance from the origin to the center of the bunch (ignoring offsets) should be the initial z distance.
        After the center is calculated, apply the x and y offsets.
        """

        # Calculate the unrotated position vector based on the initial z distance
        r_rotated = self.initial_z * np.array([np.sin(self.angle_x), np.sin(self.angle_y), 1.], dtype=np.float64)

        # Apply x and y offsets (only to r, not beta)
        r_final = r_rotated + np.array([self.offset_x, self.offset_y, 0.], dtype=np.float64)

        self.r = r_final

        # Calculate beta based on the rotated position (ignoring offsets)
        beta_direction = -r_rotated / np.linalg.norm(r_rotated)
        self.beta = beta_direction

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
                             self.angle_y, self.beta_star if self.beta_star is not None else 0)

    # def density_py(self, x, y, z):
    #     """
    #     Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
    #     :param x: float x position in lab frame
    #     :param y: float y position in lab frame
    #     :param z: float z position in lab frame
    #     :return: float Density of bunch at given point
    #     """
    #     # Broadening along the z-axis
    #     z_rel = z - self.r[2]
    #     if self.beta_star is None:
    #         sigma_x = self.sigma[0]
    #         sigma_y = self.sigma[1]
    #     else:
    #         sigma_x = self.sigma[0] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
    #         sigma_y = self.sigma[1] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
    #
    #     # Rotate coordinates in the y-z plane
    #     y_rot = (y - self.r[1]) * np.cos(self.angle_yz) - z_rel * np.sin(self.angle_yz)
    #     z_rot = (y - self.r[1]) * np.sin(self.angle_yz) + z_rel * np.cos(self.angle_yz)
    #
    #     # Calculate the density using the modified sigma_x, sigma_y, and rotated coordinates
    #     x_rel = x - self.r[0]
    #     density = np.exp(
    #         -0.5 * (x_rel ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2 + z_rot ** 2 / self.sigma[2] ** 2))
    #     density /= (2 * np.pi) ** 1.5 * sigma_x * sigma_y * self.sigma[2]  # Normalize the exponential
    #
    #     return density

    def density_py(self, x, y, z):
        """
        Calculate the density of the bunch at a given point in the lab frame, considering the y and x angles.
        The density is calculated relative to the bunch's direction, rather than the coordinate system.
        :param x: float x position in lab frame
        :param y: float y position in lab frame
        :param z: float z position in lab frame
        :return: float Density of bunch at given point
        """
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
            sigma_x = self.sigma[0]
            sigma_y = self.sigma[1]
        else:
            sigma_x = self.sigma[0] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
            sigma_y = self.sigma[1] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)

        # Calculate the density using the modified sigma_x, sigma_y, and rotated coordinates
        density = np.exp(
            -0.5 * (relative_r[0] ** 2 / sigma_x ** 2 +
                    relative_r[1] ** 2 / sigma_y ** 2 +
                    relative_r[2] ** 2 / self.sigma[2] ** 2)
        )
        density /= (2 * np.pi) ** 1.5 * sigma_x * sigma_y * self.sigma[2]  # Normalize the exponential

        return density

    def propagate(self):
        """
        Propagate the bunch to the next timestep.
        """
        self.r += self.beta * self.c * self.dt
        self.t += self.dt


# class BunchDensity:
#     """
#     Class to hold particle density for a single bunch. Currently use a 3D gaussian distribution in lab reference frame
#     (Lorentz contracted) to represent the bunch density. Use the center of the bunch as the origin of the distribution.
#     Bunch also has a 3D velocity vector and a method to calculate the density at a given point in the lab frame.
#     """
#     c = 299792458. * 1e6 / 1e9  # um/ns Speed of light
#
#     def __init__(self):
#         self.sigma = np.array([0., 0., 0.], dtype=np.float64)  # um Width of gaussian bunch profile in x, y, z
#         self.beta = np.array([0., 0., 0.], dtype=np.float64)  # v/c Dimensionless velocity of bunch in x, y, z
#         self.r = np.array([0., 0., 0.], dtype=np.float64)  # um Position of bunch center in x, y, z
#         self.t = 0.  # ns Time of bunch motion
#         self.dt = 0.  # ns Timestep to propagate bunch
#         self.angle = 0.  # Rotation angle in radians
#         self.beta_star = None  # cm Beta star value for the bunch
#
#     def set_r(self, x, y, z):
#         """
#         Set the position of the bunch.
#         :param x: float x position in lab frame
#         :param y: float y position in lab frame
#         :param z: float z position in lab frame
#         """
#         self.r = np.array([x, y, z], dtype=np.float64)
#
#     def set_beta(self, x, y, z):
#         """
#         Set the velocity of the bunch.
#         :param x: float x velocity in lab frame
#         :param y: float y velocity in lab frame
#         :param z: float z velocity in lab frame
#         """
#         self.beta = np.array([x, y, z], dtype=np.float64)
#
#     def set_sigma(self, x, y, z):
#         """
#         Set the width of the bunch in the lab frame.
#         :param x: float x width in lab frame
#         :param y: float y width in lab frame
#         :param z: float z width in lab frame
#         """
#         self.sigma = np.array([x, y, z], dtype=np.float64)
#
#     def set_angle(self, angle):
#         """
#         Set the rotation angle of the bunch in the y-z plane and adjust r and beta accordingly.
#         :param angle: float Rotation angle in radians
#         """
#         self.angle = angle
#
#         # Define rotation matrix for y-z plane
#         cos_angle = np.cos(angle)
#         sin_angle = np.sin(angle)
#
#         # Rotate position vector r
#         y_rot = self.r[1] * cos_angle + self.r[2] * sin_angle
#         z_rot = -self.r[1] * sin_angle + self.r[2] * cos_angle
#         self.r[1] = y_rot
#         self.r[2] = z_rot
#
#         # Rotate velocity vector beta
#         beta_y_rot = self.beta[1] * cos_angle + self.beta[2] * sin_angle
#         beta_z_rot = -self.beta[1] * sin_angle + self.beta[2] * cos_angle
#         self.beta[1] = beta_y_rot
#         self.beta[2] = beta_z_rot
#
#     def density(self, x, y, z):
#         """
#         Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
#         :param x: float x position in lab frame
#         :param y: float y position in lab frame
#         :param z: float z position in lab frame
#         :return: float Density of bunch at given point
#         """
#         return bdcpp.density(x, y, z, self.r[0], self.r[1], self.r[2],
#                              self.sigma[0], self.sigma[1], self.sigma[2],
#                              self.angle, self.beta_star if self.beta_star is not None else 0)
#
#     def density_py(self, x, y, z):
#         """
#         Calculate the density of the bunch at a given point in the lab frame, with broadening along z and rotation.
#         :param x: float x position in lab frame
#         :param y: float y position in lab frame
#         :param z: float z position in lab frame
#         :return: float Density of bunch at given point
#         """
#         # Broadening along the z-axis
#         z_rel = z - self.r[2]
#         if self.beta_star is None:
#             sigma_x = self.sigma[0]
#             sigma_y = self.sigma[1]
#         else:
#             sigma_x = self.sigma[0] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
#             sigma_y = self.sigma[1] * np.sqrt(1 + z ** 2 / (self.beta_star * 1e4) ** 2)
#             # sigma_x = self.sigma[0] * np.sqrt(1 + abs(z) / 1e5)
#             # sigma_y = self.sigma[1] * np.sqrt(1 + abs(z) / 1e5)
#
#         # Rotate coordinates in the y-z plane
#         y_rot = (y - self.r[1]) * np.cos(self.angle) - z_rel * np.sin(self.angle)
#         z_rot = (y - self.r[1]) * np.sin(self.angle) + z_rel * np.cos(self.angle)
#
#         # Calculate the density using the modified sigma_x, sigma_y, and rotated coordinates
#         x_rel = x - self.r[0]
#         density = np.exp(
#             -0.5 * (x_rel ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2 + z_rot ** 2 / self.sigma[2] ** 2))
#         density /= (2 * np.pi) ** 1.5 * sigma_x * sigma_y * self.sigma[2]  # Normalize the exponential
#
#         return density
#
#     def propagate(self):
#         """
#         Propagate the bunch to the next timestep.
#         """
#         self.r += self.beta * self.c * self.dt
#         self.t += self.dt
