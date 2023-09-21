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
from pyocamcalib.core.extrinsic import partial_extrinsics, get_full_rotation_matrix
from pyocamcalib.core.intrinsec import intrinsic_linear_estimate
from pyocamcalib.core._utils import check_origin, get_reprojection_error


def independente_calibration(data: dict, distortion_center: Tuple[float, float] = None):
    reprojection_error_threshold = 10.

    valid_pattern = []
    extrinsics_t = []
    taylor_coefficient_t = []
    reprojection_error_t = []

    for idx, img_path in enumerate(sorted(data.keys())):
        image_points = np.array(data[img_path]['image_points'])
        world_points = np.array(data[img_path]['world_points'])

        R_part, T_part = partial_extrinsics(image_points, world_points, img_size=(2160, 3840),
                                            distortion_center=distortion_center)

        RR = get_full_rotation_matrix(R_part, T_part, image_points, img_size=(2160, 3840),
                                      distortion_center=distortion_center)

        min_error = float('inf')
        best = None
        for r in RR:
            taylor_coefficient, r_full = intrinsic_linear_estimate(np.expand_dims(image_points, axis=0),
                                                                   np.expand_dims(world_points, axis=0),
                                                                   np.expand_dims(r, axis=0), 4,
                                                                   distortion_center=distortion_center)

            if taylor_coefficient[0] >= 0 and check_origin(r_full[:, :, :3], r_full[:, :, -1].reshape((-1, 1))):
                rms_mean, rms_std, _ = get_reprojection_error(image_points, world_points, taylor_coefficient,
                                                              np.squeeze(r_full), distortion_center)

                if rms_mean < min_error:
                    min_error = rms_mean
                    if rms_std < reprojection_error_threshold:
                        best = np.squeeze(r_full), taylor_coefficient, rms_mean

        if best is None:
            valid_pattern.append(False)
        else:
            valid_pattern.append(True)
            extrinsics_t.append(best[0])
            taylor_coefficient_t.append(best[1])
            reprojection_error_t.append(best[2])

    return valid_pattern, extrinsics_t, taylor_coefficient_t, reprojection_error_t


def get_taylor_linear(data: dict, valid: List[bool], extrinsics_t: np.array, distortion_center: Tuple[float, float]):
    image_points_t = []
    world_points_t = []
    counter = 0

    for idx, img_path in enumerate(sorted(data.keys())):
        if valid[idx]:
            image_points_t.append(np.array(data[img_path]['image_points']))
            world_points_t.append(np.array(data[img_path]['world_points']))
            counter += 1

    image_points_t = np.array(image_points_t)
    world_points_t = np.array(world_points_t)
    extrinsic_t = np.array(extrinsics_t)
    taylor_coefficient, r_full = intrinsic_linear_estimate(image_points_t,
                                                           world_points_t,
                                                           extrinsic_t, 4,
                                                           distortion_center=distortion_center)

    return taylor_coefficient, r_full


def get_first_linear_estimate(data: dict, img_size: Tuple[int, int], grid_size: int):
    initial_area = min(img_size[0] / 2, img_size[1] / 2)
    c_x, c_y = img_size[0] / 2, img_size[1] / 2
    min_rms = float("inf")
    best_d_center = (c_x, c_y)
    delta_rms = float("inf")
    counter = 0
    rms_threshold = 0.001
    best_extrinsics_t = None
    best_taylor_t = None
    valid_pattern = None

    while delta_rms > rms_threshold and counter < 10:
        ceil_size = initial_area / (2 ** (counter + 1))
        grid_x = np.linspace(best_d_center[0] - ceil_size, best_d_center[0] + ceil_size, grid_size)
        grid_y = np.linspace(best_d_center[1] - ceil_size, best_d_center[1] + ceil_size, grid_size)
        for x in grid_x:
            for y in grid_y:
                d_center = (x, y)
                valid, extrinsics_t, taylor_coefficient_t, reprojection_error_t = independente_calibration(data,
                                                                                                           distortion_center=d_center)
                overall_rms = float(np.mean(reprojection_error_t))
                if overall_rms < min_rms:
                    best_extrinsics_t = extrinsics_t
                    best_taylor_t = taylor_coefficient_t
                    valid_pattern = valid
                    delta_rms = min_rms - overall_rms
                    min_rms = overall_rms
                    best_d_center = d_center
        counter += 1
    return valid_pattern, best_d_center, min_rms, best_extrinsics_t, best_taylor_t
