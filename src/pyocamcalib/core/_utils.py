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

from typing import Tuple, List
import numpy as np
from pyocamcalib.modelling.camera import Camera


def check_origin(R: np.array, T: np.array) -> np.array:
    """
    Ambiguity on extrinsic parameters' estimation can lead to wrong origin for the camera's coordinates
    system origin. In the chessboard's coordinate system, if we name C = [Xc,Yc, Zc] the camera origin,
    we should have Zc < 0.
    How to get the camera position given the extrinsic parameters ?
    0 = RC + T => C = -inv(R) @ T  = -R.T @ T  (R is orthogonal matrix)
    :param R: Cx3x3 batch of C rotational matrix
    :param T: Cx3x1 batch of T translation vector
    :return: np.array[bool]. True if camera origin is at the right place, False otherwise.
    """
    M = -  np.transpose(R, (0, 2, 1)) @ T

    return np.squeeze(M[:, -1] < 0)


def get_reprojection_error(image_points: np.array, world_points: np.array, taylor_coefficient: np.array,
                           extrinsics: np.array, distortion_center: Tuple[float, float],
                           stretch_matrix: np.array = np.eye(2)):
    cam = Camera(taylor_coefficient, distortion_center, stretch_matrix)
    reprojected_image_points = cam.world2cam(world_points, extrinsics)
    rms_mean = np.linalg.norm(reprojected_image_points - image_points, axis=-1).mean()
    rms_std = np.linalg.norm(reprojected_image_points - image_points, axis=-1).std()

    return rms_mean, rms_std, reprojected_image_points


def get_reprojection_error_all(data: dict, valid: List[bool], extrinsics_t: np.array, taylor_coefficient: np.array,
                               distortion_center: Tuple[float, float], stretch_matrix: np.array = np.eye(2)):
    rms_mean_list = []
    rms_std_list = []
    counter = 0

    for idx, img_path in enumerate(sorted(data.keys())):
        if valid[idx]:
            image_points = np.array(data[img_path]['image_points'])
            world_points = np.array(data[img_path]['world_points'])
            extrinsics = extrinsics_t[counter]
            rms_mean, rms_std, _ = get_reprojection_error(image_points, world_points, taylor_coefficient,
                                                          extrinsics, distortion_center, stretch_matrix)
            counter += 1

            rms_mean_list.append(rms_mean)
            rms_std_list.append(rms_std)

    return np.mean(rms_mean_list), rms_mean_list, rms_std_list
