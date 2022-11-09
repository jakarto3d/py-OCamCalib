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


from typing import Union, List
import numpy as np
from scipy.optimize import least_squares
from pyocamcalib.modelling.camera import Camera


def pack(extrinsics_t_init, distortion_center_init, taylor_coefficient_init):
    x0 = []
    for i in range(extrinsics_t_init.shape[0]):
        x0.extend(list(extrinsics_t_init[i].flatten()))

    #  As a first guess for the Stretch matrix, we used the identity matrix I
    x0.extend([1, 0, 0])

    x0.extend(list(distortion_center_init))

    # Get rid of a1 coefficient which have to stay equal to 0
    ss = [taylor_coefficient_init[0], *taylor_coefficient_init[2:]]
    x0.extend(ss)

    return x0


def unpack(x, N: int):
    # Unpack flattened parameters to variables
    extrinsics_t = [np.array(x[i * 12: (i + 1) * 12]).reshape((3, 4)) for i in range(N)]
    stretch_matrix = np.array([[x[12 * N], x[12 * N + 1]],
                               [x[12 * N + 2], 1]])
    distortion_center = (x[12 * N + 3], x[12 * N + 4])
    taylor_coefficient = np.array([x[12 * N + 5], 0, *x[12 * N + 6:]])

    return extrinsics_t, stretch_matrix, distortion_center, taylor_coefficient


def bundle_adjustment_error(x: Union[np.array, float], data: dict, valid: List[bool]):
    """
    Function which computes the vector of residuals. The method jointly refines all the calibrations parameters.

    "Using the image center, together with the linear estimated parameters obtained in Section 2.2 as initial
    values is sufficient for the non-linear least squares minimization to converge
    quickly." [Improved Wide-Angle, Fisheye and Omnidirectional Camera Calibration. Steffen Urban, Jens Leitloff,
     Stefan Hinz, 2015]


    :param x: array_like with shape (n,).[flattened extrinsics parameters, flattened stretch_matrix, distortion_center,
                                          taylor_coefficient]
    This result in a vector of size n = 12 * N + 3 + 2 + taylor_order
    :param data: image_points and world_points of detected corners in chessboard pattern.
    :param valid: During the first linear calibration, not all registered pattern are considered
    (because some of them lead to a suboptimal solution). So, there is good chance that the number N of extrinsics
    matrix is inferior to the number M of pattern in data. valid is here to tell us if the pattern have to be skipped
    or not.
    :return: the L * 2-D vector of residuals. [err_x, err_y] with err_i the difference between the detected image
    points and the reprojected world points for the i coordinate.

    Notice that we only provide the vector of the residuals. scipy.optimize.least_squares constructs the cost function
    as a sum of squares of the residuals.
    """

    N = np.array(valid).sum()
    extrinsics_t, stretch_matrix, distortion_center, taylor_coefficient = unpack(x, N)
    err_w_t = []

    counter = 0
    for idx, img_path in enumerate(sorted(data.keys())):
        if valid[idx]:
            image_points = np.array(data[img_path]['image_points'])
            world_points = np.array(data[img_path]['world_points'])
            extrinsics = extrinsics_t[counter]
            cam = Camera(taylor_coefficient, distortion_center, stretch_matrix)
            reprojected_images_points = cam.world2cam(world_points, extrinsics)
            counter += 1

            err_x = image_points[:, 0] - reprojected_images_points[:, 0]
            err_y = image_points[:, 1] - reprojected_images_points[:, 1]
            err_w = np.concatenate((err_x, err_y))
            err_w_t.extend(list(err_w))

    return np.array(err_w_t)


def bundle_adjustement(data: dict, valid: List[bool], extrinsics_t_init, distortion_center_init,
                       taylor_coefficient_init):

    x0 = pack(extrinsics_t_init, distortion_center_init, taylor_coefficient_init)
    N = np.array(valid).sum()

    result = least_squares(
        bundle_adjustment_error,
        x0=x0,
        method='lm',
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        args=(data, valid)
    )

    extrinsics_t, stretch_matrix, distortion_center, taylor_coefficient = unpack(result.x, N)

    return extrinsics_t, stretch_matrix, distortion_center, taylor_coefficient
