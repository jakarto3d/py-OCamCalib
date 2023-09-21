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
import typer
import cv2 as cv
import matplotlib.pyplot as plt

from pyocamcalib.modelling.camera import Camera


def main(fisheye_image_path: str,
         calibration_file_path: str,
         perspective_fov: float,
         perspective_sensor_size: Tuple[int, int],
         ):
    """

    :param fisheye_image_path: fisheye image path.
    :param calibration_file_path: .json file with calibration parameters.
    :param perspective_fov: field of view the desired perspective camera in degree (between 0 and 180).
    :param perspective_sensor_size: (height, width) in pixels. Determine the output image resolution.
    :return:
    """

    fisheye_image = cv.imread(fisheye_image_path)
    my_camera = Camera.load_parameters_json(calibration_file_path)
    perspective_image = my_camera.cam2perspective_indirect(fisheye_image,
                                                           perspective_fov,
                                                           perspective_sensor_size)
    plt.figure()
    plt.imshow(fisheye_image[:, :, ::-1])
    plt.title('Original fisheye image')
    plt.figure()
    plt.imshow(perspective_image[:, :, ::-1])
    plt.title(f'Perspective conversion. fov = {perspective_fov} deg')
    plt.show()


if __name__ == "__main__":
    typer.run(main)
