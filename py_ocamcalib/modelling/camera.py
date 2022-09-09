"""
     Hugo Vazquez email: hugo.vazquez@jakarto.com
     Copyright (C) 2022  Hugo Vazquez

     This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License along
     with this program; if not, write to the Free Software Foundation, Inc.,
     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import json
from typing import Tuple, List, Union
import numpy as np
from modelling.utils import transform


def equirectangular2geographic(x, y):
    longitude = x * np.pi
    latitude = y * (np.pi / 2)
    return longitude, latitude


def geographic2cartesian(longitude, latitude):
    colatitude = np.pi / 2 - latitude

    Px = np.cos(latitude) * np.sin(longitude)
    Py = - np.cos(colatitude)
    Pz = np.cos(latitude) * np.cos(longitude)

    xyz_sphere = np.vstack((Px, Py, Pz)).T
    return xyz_sphere


def cartesian2geographic(xyz_points: np.array):
    latitude = - np.arctan2(xyz_points[:, 1], np.sqrt(xyz_points[:, 0] ** 2 + xyz_points[:, 2] ** 2))
    longitude = -np.arctan2(- xyz_points[:, 0], xyz_points[:, 2])

    return longitude, latitude


def normalize(r_t: np.array):
    """
    Normalize a batch of N vectors 3D vectors
    :param r_t: 3xN batch of N 3d vector
    :return: Unit vector
    """

    norm_t = np.linalg.norm(r_t, axis=1)
    r_t = (r_t.T / norm_t).T
    return r_t


def get_incident_angle(world_points: np.array):
    """
    Return the angle of incidence for each world points, i.e. the angle between the z-axis (optical axis of the camera)
    and the 3d points ray.
    :param world_points: Nx3 matrix
    :return:
    """
    world_points = normalize(world_points)
    return np.arccos(world_points[:, 2])


class Camera:
    def __init__(self,
                 taylor_coefficient: Union[np.array, List] = None,
                 distortion_center: Tuple[float, float] = None,
                 stretch_matrix: np.array = np.eye(2),
                 name: str = "myCamera",
                 inverse_poly: Union[np.array, List] = None
                 ):
        """

        :param sensor_size: size of sensor in pixel (height, width)
        :param taylor_coefficient: the polynomial mapping coefficients of the projection function [a0, a1=0, ..., aN].
        From lower to higher degree.
        :param distortion_center: the distortion vector adjusts the (0,0) location of the image plane.
        :param stretch_matrix: The stretch matrix compensates for the sensor-to-lens misalignment.
        stretch_matrix = [[c, d]
                          [e, 1]]
        """

        self.taylor_coefficient = taylor_coefficient
        self.stretch_matrix = stretch_matrix
        self.name = name
        self.distortion_center = distortion_center
        self.inverse_poly = inverse_poly

    def world2cam(self, world_points: np.array, extrinsics: np.array = None):

        """
        Map 3D world points to the sensor plane.

        :param world_points: Nx3 array of 3D points [X, Y, Z] in world's coordinates system OR in camera's coordinates
        system depending if extrinsics parameters are specified.
        :param extrinsics: The extrinsic parameters consist of a rotation, R, and a translation, t.
        The origin of the camera's coordinate system is at its optical center.
        If None, world_points are supposed to be in camera's coordinates system.

        :return: 2D points in the sensor plane.
        """

        if self.taylor_coefficient is None or self.distortion_center is None:
            raise ValueError("Fisheye parameters are empty. You first need to specify or load camera's parameters.")

        # First transform points from world's coordinates system to camera's coordinate system
        if extrinsics is not None:
            world_points = transform(extrinsics, world_points)

        # Deal with world_points = [0, 0, +-1] problem
        id_u, id_v = np.where(world_points[:, :2] == 0)
        world_points[id_u, id_v] = np.finfo(float).eps

        nb_points = world_points.shape[0]
        world_radius = np.sqrt(world_points[:, 0] ** 2 + world_points[:, 1] ** 2)
        z_scaled = world_points[:, 2] / world_radius

        rho = []
        for i in range(nb_points):
            taylor_tmp = self.taylor_coefficient[::-1].copy()
            taylor_tmp[-2] -= z_scaled[i]
            roots = np.roots(taylor_tmp)
            roots = roots[(roots > 0) & (np.imag(roots) == 0)]
            roots = np.array([float(np.real(e)) for e in roots])
            if roots.shape[0] == 0:
                rho.append(np.nan)
            else:
                rho.append(np.min(roots))

        rho = np.array(rho)

        # Get 2D points in image plane
        image_points_x = (world_points[:, 0] / world_radius) * rho
        image_points_y = (world_points[:, 1] / world_radius) * rho

        # Get 2D points in image sensor
        image_points_x = image_points_x * self.stretch_matrix[0, 0] + image_points_y * self.stretch_matrix[0, 1]
        image_points_y = image_points_x * self.stretch_matrix[1, 0] + image_points_y

        image_points_x += self.distortion_center[0]
        image_points_y += self.distortion_center[1]

        image_points = np.vstack((image_points_x, image_points_y)).T

        return image_points

    def cam2world(self, uv_points: np.array):

        """
        Given an image point it returns the 3D coordinates of its correspondent optical
        ray on the unit sphere.
        :param uv_points: Nx2 array which contains coordinates of N pixels (origin at top left corner).
        :return: Nx3 array which contains corresponding unit vector on the unit sphere.
        """
        if self.taylor_coefficient is None or self.distortion_center is None:
            raise ValueError("Fisheye parameters are empty. You first need to specify or load camera's parameters.")

        # First transform the sensor pixel point to the ideal image pixel point ().
        uv_points -= self.distortion_center
        stretch_inv = np.linalg.inv(self.stretch_matrix)
        uv_points = uv_points @ stretch_inv.T

        rho = np.sqrt(uv_points[:, 0] ** 2 + uv_points[:, 1] ** 2)
        x = uv_points[:, 0]
        y = uv_points[:, 1]
        z = np.polyval(self.taylor_coefficient[::-1], rho)
        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        world_points = np.vstack((x, y, z)).T / norm[:, None]

        return world_points

    def world2cam_fast(self, world_points: np.array, extrinsics: np.array = None):
        """
        Map 3D world points to the sensor plane using polynomial approximation. This is really fast compare to original
        function.

        :param world_points: Nx3 array of 3D points [X, Y, Z] in world's coordinates system OR in camera's coordinates
        system depending if extrinsics parameters are specified.
        :param extrinsics: The extrinsic parameters consist of a rotation, R, and a translation, t.
        The origin of the camera's coordinate system is at its optical center.
        If None, world_points are supposed to be in camera's coordinates system.

        :return: 2D points in the sensor plane.
        """
        if self.inverse_poly is None:
            raise ValueError("You first need to fond polynomial approximation of inverse function.")

            # First transform points from world's coordinates system to camera's coordinate system
        if extrinsics is not None:
            world_points = transform(extrinsics, world_points)

        theta = get_incident_angle(world_points)
        rho = np.polyval(self.inverse_poly, theta)
        world_radius = np.sqrt(world_points[:, 0] ** 2 + world_points[:, 1] ** 2)

        # Get 2D points in image plane
        image_points_x = (world_points[:, 0] / world_radius) * rho
        image_points_y = (world_points[:, 1] / world_radius) * rho

        # Get 2D points in image sensor
        image_points_x = image_points_x * self.stretch_matrix[0, 0] + image_points_y * self.stretch_matrix[0, 1]
        image_points_y = image_points_x * self.stretch_matrix[1, 0] + image_points_y

        image_points_x += self.distortion_center[0]
        image_points_y += self.distortion_center[1]

        image_points = np.vstack((image_points_x, image_points_y)).T

        return image_points

    def undistort(self):
        pass

    def load_parameters(self, parameters_path: str):
        """
        Load calibration parameters from .json file.
        :param parameters_path: path to .json file.
        :return:
        """
        with open(parameters_path, 'r') as f:
            calib = json.load(f)

        self.distortion_center = np.array(calib["distortion_center"])
        self.stretch_matrix = np.array(calib["stretch_matrix"])
        self.taylor_coefficient = np.array(calib["taylor_coefficient"])
        self.inverse_poly = np.array(calib["inverse_poly"])

    def cam2equirectangular(self, fisheye_image: np.array, extrinsic: np.array, equirectangular_size: Tuple[int, int]):
        """
        Map a fisheye image to the equirectangular plane. Here, the z-axis of the camera's coordinate system point in
        the direction of the optical axis.
        This function use inverse warping in order to avoid interpolation process. This mean that for each pixel of the
        equirectangular plane, we are looking for the corresponding pixel in the fisheye image.
        :param extrinsic: Specify rotation and translation from camera's coordinate system to world before reproject
        point on the equirectangular plane.
        :param fisheye_image: fisheye image as numpy array
        :param equirectangular_size: size of the desired equirectangular plane.
        :return: equirectangular image as numpy array
        """

        # Fisheye parameters
        hf, wf, n_band = fisheye_image.shape

        # Create equirectangular image
        he, we = equirectangular_size
        equirectangular_img = np.zeros((he, we, n_band)).astype(np.uint8)
        grid = np.indices((he, we))
        xe = grid[1].flatten()
        ye = grid[0].flatten()

        # Normalize coordinates
        xe_n = (2 * xe - we) / we
        ye_n = (- 2 * ye + he) / he

        # Compute coordinates in the fisheye projection
        longitude, latitude = equirectangular2geographic(xe_n, ye_n)
        xyz_sphere = geographic2cartesian(longitude, latitude)
        uv_points = self.world2cam_fast(xyz_sphere, extrinsic)
        uv_points = np.round(uv_points).astype(int)
        cond_1 = (uv_points[:, 1] < hf) & (uv_points[:, 1] > 0)
        cond_2 = (uv_points[:, 0] < wf) & (uv_points[:, 0] > 0)
        mask = cond_1 & cond_2
        equirectangular_img[ye[mask], xe[mask]] = fisheye_image[uv_points[:, 1][mask], uv_points[:, 0][mask]]

        return equirectangular_img
