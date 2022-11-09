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
from pyocamcalib.core.linear_estimation import independente_calibration


def find_center(data: dict, img_size: Tuple[int, int], grid_size: int, initial_ssre: float):
    initial_area = min(img_size[0] / 2, img_size[1] / 2)
    c_x, c_y = img_size[1] / 2, img_size[0] / 2
    min_ssre = initial_ssre
    best_d_center = (c_x, c_y)
    delta_ssre = float("inf")
    counter = 0
    ssre_threshold = 0.001
    best_results = None

    while delta_ssre > ssre_threshold and counter < 10:
        ceil_size = initial_area / (2 ** (counter + 1))
        grid_x = np.linspace(best_d_center[0] - ceil_size, best_d_center[0] + ceil_size, grid_size)
        grid_y = np.linspace(best_d_center[1] - ceil_size, best_d_center[1] + ceil_size, grid_size)
        for x in grid_x:
            for y in grid_y:
                d_center = (x, y)
                valid, results = independente_calibration(data, distortion_center=d_center)
                ssre = float(np.mean([e[-1] for e in results]))
                if ssre < min_ssre:
                    best_results = results
                    delta_ssre = min_ssre - ssre
                    min_ssre = ssre
                    best_d_center = d_center
        counter += 1
    return best_d_center, min_ssre, best_results
