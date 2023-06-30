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
