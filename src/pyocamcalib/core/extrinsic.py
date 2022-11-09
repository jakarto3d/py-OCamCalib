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

from typing import Tuple
import numpy as np


def partial_extrinsics(image_points: np.array, world_points: np.array, img_size: Tuple[int, int],
                       distortion_center: Tuple[float, float] = None) -> np.array:
    """
    The partial extrinsics parameters stack in the vector H are estimated by minimizing
    the least-squares criterion ||MH||^2 subject to ||H||^2 = 1. This is accomplished by
    using the Singular Value Decomposition (SVD).
    The solution is known up to a scale factor which can be determined uniquely since R1, R2
    are orthonormal.
    Parameters t3 will be determined during the estimation of the intrinsics parameters.

    :param img_size: (height, width) in pixel
    :param image_points: Nx2 matrix with coordinates of corner in fisheye image
    :param world_points: Nx2 matrix with coordinates of corner in 3D world
    :return: partial extrinsics parameters H = [r11, r21, r21, r22, t1, t2]
    :param distortion_center:
    """
    image_points_c = image_points.copy()
    if distortion_center:
        image_points_c[:, 0] -= distortion_center[0]
        image_points_c[:, 1] -= distortion_center[1]
    else:
        center_x, center_y = img_size[1] / 2, img_size[0] / 2
        image_points_c[:, 0] -= center_x
        image_points_c[:, 1] -= center_y

    M = np.stack([- image_points_c[:, 1] * world_points[:, 0],
                  - image_points_c[:, 1] * world_points[:, 1],
                  + image_points_c[:, 0] * world_points[:, 0],
                  + image_points_c[:, 0] * world_points[:, 1],
                  - image_points_c[:, 1],
                  + image_points_c[:, 0]], axis=1)
    u, s, vh = np.linalg.svd(M, full_matrices=False)

    # Get the last row of Vh, i.e. (the last column of V)
    H = vh[-1, :]

    R_part = np.array([[H[0], H[1]], [H[2], H[3]]])
    T_part = np.array([H[4], H[5]])

    return R_part, T_part


def get_full_rotation_matrix(R_part: np.array, T_part: np.array, image_points, img_size: Tuple[int, int],
                             distortion_center: Tuple[float, float] = None):
    """
    Given the scaled `2 x 2` submatrix of a `3 x 3` rotation matrix
    and the scaled `(x, y)` components of a 3d translation vector
    solve for the full `3 x 3` rotation matrix `R` and the scale factor.

    :param image_points:
    :param R_part: 2x2 scaled matrix which contains pat of the full rotational matrix
    :param T_part: 2x1 acaled vector which contains part of the translation vector
    :return: All solution for unscaled 3x3 rotational matrix and translation vector
    :param distortion_center:
    """
    image_points_c = image_points.copy()
    if distortion_center:
        image_points_c[:, 0] -= distortion_center[0]
        image_points_c[:, 1] -= distortion_center[1]
    else:
        center_x, center_y = img_size[1] / 2, img_size[0] / 2
        image_points_c[:, 0] -= center_x
        image_points_c[:, 1] -= center_y

    A = (R_part[0, 0] * R_part[0, 1] + R_part[1, 0] * R_part[1, 1]) ** 2
    B = R_part[0, 0] ** 2 + R_part[1, 0] ** 2
    C = R_part[0, 1] ** 2 + R_part[1, 1] ** 2

    r32_square_all = np.roots([1, C - B, -A])
    r32_square_all = r32_square_all[r32_square_all >= 0]

    r_31_all = []
    r_32_all = []
    sg = [-1, 1]

    for i in range(r32_square_all.shape[0]):
        for j in range(2):
            r_32 = sg[j] * np.sqrt(r32_square_all[i])
            r_32_all.append(r_32)
            if r_32 ** 2 <= 0.00000001 * (R_part[1, 0] ** 2 + R_part[1, 1] ** 2):
                r_31_all.append(np.sqrt(C - B))
                r_31_all.append(- np.sqrt(C - B))
                r_32_all.append(r_32)
            else:
                r_31_all.append(-(R_part[0, 0] * R_part[0, 1] + R_part[1, 0] * R_part[1, 1]) / r_32)

    RR = np.zeros((2 * len(r_32_all), 3, 3))
    count = 0
    for i in range(len(r_32_all)):
        for j in range(2):
            scale = 1 / np.sqrt(B + r_31_all[i] ** 2)
            RR[count, :, :] = sg[j] * scale * np.array([[R_part[0, 0], R_part[0, 1], T_part[0]],
                                                        [R_part[1, 0], R_part[1, 1], T_part[1]],
                                                        [r_31_all[i], r_32_all[i], 0]])
            count += 1

    #  Construct the 3rd column of `3 x 2` sub matrix of a `3 x 3` rotation matrix
    RR_new = []
    for idx, r in enumerate(RR):
        r3 = np.cross(r[:, 0], r[:, 1])
        norm_r3 = np.linalg.norm(r3)
        r3 /= norm_r3
        RR_new.append(np.insert(r, 2, r3, axis=1))

    return np.array(RR_new)


def get_full_rotation_matrix_v2(R_part: np.array, T_part: np.array, image_points, img_size: Tuple[int, int]):
    """
    Given the scaled `2 x 2` submatrix of a `3 x 3` rotation matrix
    and the scaled `(x, y)` components of a 3d translation vector
    solve for the full `3 x 3` rotation matrix `R` and the scale factor.

    :param R_part: 2x2 scaled matrix which contains pat of the full rotational matrix
    :param T_part: 2x1 acaled vector which contains part of the translation vector
    :return: Full unscaled 3x3 rotational matrix, unscaled partial translation vector, scale factor
    """
    center_x, center_y = img_size[1] / 2, img_size[0] / 2
    image_points_c = image_points.copy()
    image_points_c[:, 0] -= center_x
    image_points_c[:, 1] -= center_y

    A = (R_part[0, 0] * R_part[0, 1] + R_part[1, 0] * R_part[1, 1]) ** 2
    B = R_part[0, 0] ** 2 + R_part[1, 0] ** 2
    C = R_part[0, 1] ** 2 + R_part[1, 1] ** 2
    #
    delta = (C - B) ** 2 + 4 * A
    p = C - B
    d = np.sqrt(delta)

    r_31_all = [+ np.sqrt((p + d) / 2),
                - np.sqrt((p + d) / 2)]

    if p - d > 0:
        r_31_all.extend([+ np.sqrt((p - d) / 2), - np.sqrt((p - d) / 2)])

    scale_r31 = [1 / (np.sqrt(R_part[0, 0] ** 2 + R_part[1, 0] ** 2 + r_31_all[i] ** 2)) for i in [0, 2] if
                 i < len(r_31_all)]

    r_32_all = [np.sqrt(1 / scale ** 2 - R_part[0, 1] ** 2 - R_part[1, 1] ** 2) for scale in scale_r31]
    r_32_all.extend([-e for e in r_32_all])
    scale_r31.extend([-e for e in scale_r31])


