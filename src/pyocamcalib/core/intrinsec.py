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

import numpy as np


def intrinsic_linear_estimate(image_points_t: np.array, world_points_t: np.array, extrinsic_t: np.array,
                              taylor_order: int, distortion_center: (int, int) = None):
    """
    First linear estimation of the intrinsec parameters [a0, a1, ..., aN] ans last translation coordinates
    [t1, t2, ..., tC] by pseudo-inverse matrix method.


    :param taylor_order: degree of the polynomial use for the Taylor imaging model.
    :param image_points_t: CxNx2 array which contains raw coordinates of corners in image frame, with C
    the number of pattern captured and N the number of corners by pattern.
    :param world_points_t: CxNx2 array which contains coordinates of corners in world frame, i.e. (0,0), (0,1),
    (0,2), ... if calibration is done in chessboard square unit or (0,30), (0,60), (0,90), ... if calibration is
    done in mm unit with square side of 30 mm for instance. C is the number of pattern captured and N the number of
    corners by pattern.
    :param extrinsic_t: Cx3x4 array which contains the first linear estimate of camera extrinsic parameters. We got
    3x3 matrix for each C pattern captured.
    :param distortion_center:
    :return: The intrinsec parameters and last translation coordinates [a0, a1, ..., aN, t3_1, t3_2, t3_C], with N
    the degree of the best polynomial degree.
    """

    nb_pattern = extrinsic_t.shape[0]
    nb_points = image_points_t.shape[1]

    image_points_t_c = image_points_t.copy()

    image_points_t_c[:, :, 0] -= distortion_center[0]
    image_points_t_c[:, :, 1] -= distortion_center[1]

    A = extrinsic_t[:, 1, 0][:, None] * world_points_t[:, :, 0] + \
        extrinsic_t[:, 1, 1][:, None] * world_points_t[:, :, 1] + \
        extrinsic_t[:, 1, 3][:, None] * np.ones((nb_pattern, nb_points))

    B = image_points_t_c[:, :, 1] * (extrinsic_t[:, 2, 0][:, None] * world_points_t[:, :, 0] +
                                     extrinsic_t[:, 2, 1][:, None] * world_points_t[:, :, 1])

    C = extrinsic_t[:, 0, 0][:, None] * world_points_t[:, :, 0] + \
        extrinsic_t[:, 0, 1][:, None] * world_points_t[:, :, 1] + \
        extrinsic_t[:, 0, 3][:, None] * np.ones((nb_pattern, nb_points))

    D = image_points_t_c[:, :, 0] * (extrinsic_t[:, 2, 0][:, None] * world_points_t[:, :, 0] +
                                     extrinsic_t[:, 2, 1][:, None] * world_points_t[:, :, 1])

    rho = np.sqrt(image_points_t_c[:, :, 0] ** 2 + image_points_t_c[:, :, 1] ** 2).flatten()
    M_rho = np.power(rho.reshape((rho.shape[0], 1)), np.arange(taylor_order + 1).reshape((1, taylor_order + 1)))

    # Delete second column of M-rho because coefficient a1 = 0
    M_rho = np.delete(M_rho, 1, 1)

    sub_M_1 = A.flatten()[:, None] * M_rho
    sub_M_2 = C.flatten()[:, None] * M_rho

    sub_M_3 = np.zeros((nb_pattern * nb_points, nb_pattern))
    sub_M_4 = np.zeros((nb_pattern * nb_points, nb_pattern))

    for i in range(nb_pattern):
        sub_M_3[i * nb_points:(i + 1) * nb_points, i] = - image_points_t_c[i, :, 1]

    for i in range(nb_pattern):
        sub_M_4[i * nb_points:(i + 1) * nb_points, i] = - image_points_t_c[i, :, 0]

    M_1 = np.concatenate((sub_M_1, sub_M_2), axis=0)
    M_2 = np.concatenate((sub_M_3, sub_M_4), axis=0)
    M = np.concatenate((M_1, M_2), axis=1)

    E = np.hstack((B.flatten(), D.flatten()))
    s = np.linalg.pinv(M) @ E
    taylor_coefficient = [s[0], 0, *s[1:taylor_order]]
    t3_t = s[taylor_order:]

    extrinsic_t[:, 2, 3] = t3_t

    return taylor_coefficient, extrinsic_t
