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
from datetime import date
from typing import Tuple, List
import cv2 as cv
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.PNG']


def transform(extrinsics: np.array, world_points: np.array):
    """
    Transform D world points from world coordinates to camera's coordinate system.
    :param extrinsics: 3x4 array which consist of a rotation, R, and a translation, t. [R|t]
    :param world_points: Nx3 array of world points in world coordinates [X, Y, Z]
    :return: Nx3 array of 3D points in camera's coordinate system
    """
    rotation_matrix = extrinsics[:, :3]
    translation_vector = extrinsics[:, -1]
    world_points_c = world_points @ rotation_matrix.T
    world_points_c += translation_vector

    return world_points_c


def has_file_allowed_extension(filename):
    """
    Checks if a file is an allowed extension.

    :param filename: path to a file
    :return: bool. True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_files(path):
    """
    Get all files in a folder that respect allowed extensions
    :param path: working_dir
    :return: all files with allowed extensions
    """
    all_files = []
    for ext in IMG_EXTENSIONS:
        all_files.extend(list(path.glob("*" + ext)))
    return all_files


def generate_checkerboard_points(board_size: Tuple[int, int], square_size: float = 1, z_axis: bool = False):
    """
    Generate checkerboard corner coordinates in world's coordinate system.
    The point [0, 0, 0] corresponds to the lower-right corner of the top-left square of the board.
    The number of points, N, that the function returns are based on the number of squares on the checkerboard.
    :param board_size: Number of inner square : (along the x_direction, along the y_direction)
    :param square_size: i.e. (0,0), (0,1), (0,2), ... if calibration is done in chessboard square unit or
    (0,30), (0,60), (0,90), ... if calibration is done in mm unit with square side of 30 mm for instance.
    :param z_axis: if True world_points with z-coordinates otherwise only x-y-coordinates.
    :return:
    """
    # 3D points are called object points , like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    world_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    world_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    world_points *= square_size
    if z_axis:
        return world_points
    else:
        return world_points[:, :2]


def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        if params[1]:
            params[5].append([x, y])
            cv.drawMarker(params[0], (x, y), (255, 0, 0))
            cv.imshow('image', params[0])
            params[1] = False


def get_selected_corners(x_0, y_0, width, height, corners: np.array):
    nb_selected = x_0.shape[0]
    idx = []
    for i in range(nb_selected):
        cond_1 = x_0[i] <= corners[:, 0]
        cond_2 = corners[:, 0] < x_0[i] + width[i]
        cond_3 = y_0[i] <= corners[:, 1]
        cond_4 = corners[:, 1] < y_0[i] + height[i]
        mask = cond_1 & cond_2 & cond_3 & cond_4
        if mask.sum() > 0:
            idx.append(int(np.where(mask)[0]))
    return idx


def write_text(image: np.array, text: str):
    y_start = 150
    y_increment = 100
    for i, line in enumerate(text.split('\n')):
        y = y_start + i * y_increment
        cv.putText(img=image, text=line, org=(150, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                   color=(76, 153, 0), thickness=3)


def check_detection(corners, image):
    """
    Allows to check corners that were detected automatically and to modify some of them if necessary.
    You can enter in two different mode : SELECTION MODE and DRAW MODE.
    SELECTION MODE: Press 's' to enter the mode. Selection mode allows you to select points that have not been detected
    accurately. After pressing 's' you can surround such point with bounding boxe by pressing left mouse, drawing
    bounding boxes AND confirm by pressing enter. Once you select all your points, you have to quit selection mode by
    pressing 'esc'. Selected points should appear RED.
    DRAW MODE: Press 'd' each time you want draw the new point and click to drop off your modified points. New point
    have to be drawn in same order then selected points.

    When work is done, press 'z' to quit. A windows with your modified pattern should appears.
    :param corners:
    :param image:
    :return:
    """
    mode_draw = False
    mode_select = False
    wait = False
    new_corners = []
    idx = []

    text_1 = "Press:\n's' select mode : zoom out,  draw bbox with mouse, 'enter' to validate, 'esc' to end " \
             "\n'd' to draw point\n'z' to quit"

    text_2 = "Here is the modified pattern !"

    image_draw = image.copy()
    write_text(image_draw, text_1)
    for corner in corners:
        cv.drawMarker(image_draw, tuple(corner.astype(int)), (0, 255, 0))
    cv.namedWindow("image", 2)
    cv.imshow('image', image_draw)
    params = [image_draw, mode_draw, mode_select, corners, idx, new_corners, wait]
    cv.setMouseCallback('image', click_event, params)

    while 1:
        k = cv.waitKey(1) & 0xFF

        # press 's' to enter in select mode (i.e. select point that had been badly detected)
        if k == ord('s'):
            roi = cv.selectROIs('image', image_draw)
            if isinstance(roi, np.ndarray):
                width = roi[:, 2]
                height = roi[:, 3]
                x_0 = roi[:, 0]
                y_0 = roi[:, 1]
                index = get_selected_corners(x_0, y_0, width, height, corners)
                params[4].extend(index)
                for _id in index:
                    cv.drawMarker(image_draw, tuple(corners[_id].astype(int)), (0, 0, 255))
                cv.imshow('image', image_draw)
                cv.setMouseCallback('image', click_event, params)

        # press 'd' to select a new point
        if k == ord('d'):
            params[1] = True

        # press 'z' to quit
        if k == ord('z'):
            break
    cv.destroyAllWindows()

    if (len(idx) == len(new_corners)) and len(idx) != 0:
        corners[idx] = new_corners
        cv.namedWindow("Modified pattern", 2)
        image_draw_2 = image.copy()
        write_text(image_draw_2, text_2)
        for corner in corners:
            cv.drawMarker(image_draw_2, tuple(corner.astype(int)), (0, 255, 0))
        cv.imshow('Modified pattern', image_draw_2)
        cv.waitKey(0)
        cv.destroyAllWindows()


def normalize(r_t: np.array):
    """
    Normalize a batch of N vectors 3D vectors
    :param r_t: 3xN batch of N 3d vector
    :return: Unit vector
    """

    norm_t = np.linalg.norm(r_t, axis=1)
    r_t = (r_t.T / norm_t).T
    return r_t


def proj(u_t: np.array, v_t: np.array):
    """
    Projection operator function. Project the vector v on the vector u
    :param u_t:
    :param v_t:
    :return:
    """
    dot_uv = np.repeat(np.sum(u_t * v_t, axis=1).reshape((-1, 1)), u_t.shape[1], axis=1)
    dot_uu = np.repeat(np.sum(u_t * u_t, axis=1).reshape((-1, 1)), u_t.shape[1], axis=1)
    return (dot_uv / dot_uu) * u_t


def gram_schmidt(R_t: np.array) -> np.array:
    """
    The Gram–Schmidt process is a method for orthonormalizing a set of vectors
     in an inner product space
    :param R_t: Cx3xN batch of C set of 3D vectors, each set containing N vectors
    :return: C batch of N 3D vectors orthonormalized
    """
    N = R_t.shape[-1]
    R_t_new = []
    for i in range(N):
        r = R_t[:, :, i]
        r_aux = r.copy()
        for j in range(i):
            r -= proj(R_t_new[j], r_aux)
        R_t_new.append(normalize(r))

    R_t_new = np.swapaxes(R_t_new, 0, 1)
    R_t_new = np.swapaxes(R_t_new, 1, 2)
    return np.array(R_t_new)


def get_incident_angle(world_points: np.array):
    """
    Return the angle of incidence for each world points, i.e. the angle between the z-axis (optical axis of the camera)
    and the 3d points ray.
    :param world_points: Nx3 matrix
    :return:
    """
    world_points = normalize(world_points)
    return np.arccos(world_points[:, 2])


def get_canonical_projection_model(model_name: str, fov: float):
    if model_name == "rectilinear":
        theta = np.linspace(0, np.radians(90))
        r = np.tan(theta)
        return r, np.degrees(theta)

    if model_name == "equidistant":
        theta = np.linspace(0, np.radians(fov / 2))
        f = 1 / (np.radians(fov) / 2)
        r = f * theta
        return r, np.degrees(theta)

    if model_name == "equisolid":
        theta = np.linspace(0, np.radians(fov / 2))
        f = 1 / (2 * np.sin(np.radians(fov) / 4))
        r = 2 * f * np.sin(theta / 2)
        return r, np.degrees(theta)

    if model_name == "stereographic":
        theta = np.linspace(0, np.radians(fov / 2))
        f = 1 / (2 * np.tan(np.radians(fov) / 4))
        r = 2 * f * np.tan(theta / 2)
        return r, np.degrees(theta)


def save_calib(valid: List[bool], extrinsics_t: np.array, img_path: List[str],
               taylor_coefficient: np.array, distortion_center: Tuple[float, float],
               stretch_matrix: np.array, camera_name: str, rms_overall: float, rms_mean_list: List[float],
               rms_std_list: float):
    """
    Write calibration results in .json file
    :param rms_std_list:
    :param rms_mean_list:
    :param rms_overall:  Average rms on all valid pattern
    :param stretch_matrix:
    :param camera_name:
    :param taylor_coefficient:
    :param img_path:
    :param extrinsics_t:
    :param valid:
    :param distortion_center: (float, float)
    :return: 0
    """
    today = date.today()
    time = today.strftime("%d/%m/%Y")
    outputs = {"date": time,
               "camera_name": camera_name,
               "valid": valid,
               "taylor_coefficient": taylor_coefficient.tolist(),
               "distortion_center": distortion_center,
               "stretch_matrix": stretch_matrix.tolist(),
               "extrinsics_t": [e.tolist() for e in extrinsics_t],
               "img_path": img_path,
               "rms_overall": rms_overall,
               "rms_mean_list": rms_mean_list,
               "rms_std_list": rms_std_list
               }

    with open('calib_opt.json', 'w') as f:
        json.dump(outputs, f, indent=4)


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
