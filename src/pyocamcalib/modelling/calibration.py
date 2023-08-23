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
import pickle
from itertools import product
from pathlib import Path
from typing import Tuple
import cv2 as cv
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger

from pyocamcalib.core._utils import get_reprojection_error_all, get_reprojection_error
from pyocamcalib.core.linear_estimation import get_first_linear_estimate, get_taylor_linear
from pyocamcalib.core.optim import bundle_adjustement
from pyocamcalib.modelling.utils import get_files, generate_checkerboard_points, check_detection, transform, save_calib, \
    get_canonical_projection_model, Loader, get_incident_angle


class CalibrationEngine:
    def __init__(self,
                 working_dir: str,
                 chessboard_size: Tuple[int, int],
                 camera_name: str,
                 square_size: float = 1):
        """
        :param working_dir: path to folder which contains all chessboard images
        :param chessboard_size: Number of INNER corners per a chessboard (row, column)
        """
        self.rms_std_list = None
        self.rms_mean_list = None
        self.rms_overall = None
        self.extrinsics_t_linear = None
        self.taylor_coefficient_linear = None
        self.working_dir = Path(working_dir)
        self.images_path = [str(e) for e in get_files(Path(working_dir))]
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.sensor_size = cv.imread(str(self.images_path[0])).shape[:2][::-1]
        self.distortion_center = (self.sensor_size[0] / 2, self.sensor_size[1] / 2)
        self.detections = {}
        self.distortion_center_linear = None
        self.extrinsics_t = None
        self.taylor_coefficient = None
        self.stretch_matrix = None
        self.valid_pattern = None
        self.cam_name = camera_name
        self.inverse_poly = None

    def detect_corners(self, check: bool = False, max_height: int = 520):
        images_path = get_files(self.working_dir)
        count = 0
        world_points = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=True)

        logger.info("Start corners extraction")

        for img_f in tqdm(sorted(images_path)):
            img = cv.imread(str(img_f))
            height, width = img.shape[:2]
            ratio = width / height
            img_resize = cv.resize(img, (round(ratio * max_height), max_height))
            r_h = height / max_height
            r_w = width / (ratio * max_height)

            gray_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            for block, bias in list(product(range(20, 40, 5), range(-10, 31, 5))):

                block = (block // 2) * 2 + 1
                img_bw = cv.adaptiveThreshold(gray_resize, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block,
                                              bias)
                ret, corners = cv.findChessboardCornersSB(img_bw, self.chessboard_size, flags=cv.CALIB_CB_EXHAUSTIVE)

                if not ret:
                    ret, corners = cv.findChessboardCornersSB(img_bw, self.chessboard_size, flags=0)

                if ret:
                    corners = np.squeeze(corners)
                    corners[:, 0] *= r_w
                    corners[:, 1] *= r_h
                    win_size = (5, 5)
                    zero_zone = (-1, -1)
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
                    corners = np.expand_dims(corners, axis=0)
                    cv.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                    if check:
                        check_detection(np.squeeze(corners), img)
                    count += 1
                    self.detections[str(img_f)] = {"image_points": np.squeeze(corners)[::-1],
                                                   "world_points": np.squeeze(world_points)}
                    break

        logger.info(f"Extracted chessboard corners with success = {count}/{len(images_path)}")

    def save_detection(self):
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        with open(f'./../checkpoints/corners_detection/detections_{self.cam_name}_{dt_string}.pickle',
                  'wb') as f:
            pickle.dump(self.detections, f)

            logger.info(f"Detection file saved with success.")

    def load_detection(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.detections = pickle.load(f)
            logger.info("Detection file loaded with success.")

    def estimate_fisheye_parameters(self, grid_size: int = 5):
        if not self.detections:
            raise ValueError('Detections is empty. You first need to detect corners in several chessboard images or '
                             'load a detection file.')

        loader = Loader("INFO:: Start first linear estimation ... ", "", 0.5).start()
        valid_pattern, d_center, min_rms, extrinsics_t, taylor_t = get_first_linear_estimate(self.detections,
                                                                                             self.sensor_size,
                                                                                             grid_size)

        taylor_coefficient, extrinsics_t = get_taylor_linear(self.detections, valid_pattern, extrinsics_t, d_center)
        loader.stop()
        rms_overall, _, _ = get_reprojection_error_all(self.detections, valid_pattern,
                                                       extrinsics_t, taylor_coefficient,
                                                       d_center)

        logger.info(f"Linear estimation end with success \n"
                    f"Linear RMS = {rms_overall:0.2f} \n"
                    f"Distortion Center = {d_center}\n"
                    f"Taylor_coefficient = {taylor_coefficient}\n")

        self.distortion_center_linear = d_center
        self.taylor_coefficient_linear = taylor_coefficient
        self.extrinsics_t_linear = extrinsics_t
        self.valid_pattern = valid_pattern

        loader = Loader("INFO:: Start bundle adjustment  ... ", "", 0.5).start()
        extrinsics_t_opt, stretch_matrix, d_center_opt, taylor_coefficient_opt = bundle_adjustement(self.detections,
                                                                                                    valid_pattern,
                                                                                                    extrinsics_t,
                                                                                                    d_center,
                                                                                                    taylor_coefficient)
        loader.stop()
        self.distortion_center = d_center_opt
        self.taylor_coefficient = taylor_coefficient_opt
        self.extrinsics_t = extrinsics_t_opt
        self.stretch_matrix = stretch_matrix

        rms_overall, rms_mean_list, rms_std_list = get_reprojection_error_all(self.detections, valid_pattern,
                                                                              self.extrinsics_t,
                                                                              self.taylor_coefficient,
                                                                              self.distortion_center,
                                                                              self.stretch_matrix)

        logger.info(f"Bundle Adjustment end with success \n"
                    f"Optimize rms = {rms_overall:0.2f} \n"
                    f"Distortion Center = {d_center_opt}\n"
                    f"Taylor_coefficient = {taylor_coefficient_opt}\n")

        self.rms_overall = rms_overall
        self.rms_mean_list = rms_mean_list
        self.rms_std_list = rms_std_list

        # save_calib(valid_pattern, extrinsics_t_opt, self.images_path,
        #            self.taylor_coefficient, self.distortion_center,
        #            self.stretch_matrix, self.cam_name, rms_overall, rms_mean_list, rms_std_list)

    def get_chessboard_position(self, save: bool = False):
        """
        Write the four vertices of each chessboard in camera's coordinate system.
        :param save:
        :return: x
        """
        if not self.detections:
            raise ValueError(
                'Detections is empty. You first need to detect corners in several chessboard images or '
                'load a detection file.')

        if not self.extrinsics_t:
            raise ValueError('Extrinsics parameters are empty. You first need to calibrate camera.')

        world_points = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=True)
        world_points = world_points[[0,
                                     self.chessboard_size[0] * (self.chessboard_size[1] - 1),
                                     self.chessboard_size[0] * self.chessboard_size[1] - 1,
                                     self.chessboard_size[0] - 1], :]
        world_points_c = []

        for r in self.extrinsics_t:
            world_points_c.append(transform(r, world_points).tolist())

        if save:
            with open('./../checkpoints/chessboard_position.json', 'w') as f:
                json.dump(world_points_c, f, indent=4)

        return world_points_c

    def show_reprojection_error(self, save: bool = True):

        plt.figure(figsize=(20, 20))
        plt.bar(np.arange(len(self.rms_mean_list)), self.rms_mean_list, yerr=self.rms_std_list, align='center',
                alpha=0.5, ecolor='black', capsize=10)
        plt.axhline(self.rms_overall, color='g', linestyle='--', label=f"Overall RMS = {self.rms_overall:0.2f}")
        plt.ylabel('Mean Error in Pixels', fontsize=15)
        plt.xlabel("Images", fontsize=15)
        plt.title(f'Mean Reprojection Error per Image {self.cam_name}', fontsize=20)
        plt.legend()
        if save:
            plt.savefig(f"./../../../docs/Mean_reprojection_error_{self.cam_name}.png", dpi=300)
        plt.show()

    def show_reprojection(self):

        if not self.detections:
            raise ValueError(
                'Detections is empty. You first need to detect corners in several chessboard images or '
                'load a detection file.')

        if self.extrinsics_t is None or self.taylor_coefficient is None:
            raise ValueError(
                'Camera parameters are empty. You first need to perform calibration or load calibration file.')

        counter = 0
        for id_im, img_path in enumerate(sorted(self.detections.keys())):
            if self.valid_pattern[id_im]:
                image_points = np.array(self.detections[img_path]['image_points'])
                world_points = np.array(self.detections[img_path]['world_points'])
                extrinsics = self.extrinsics_t[counter]

                im = cv.imread(str(img_path))
                h, w = im.shape[:2]
                im_center = (w / 2, h / 2)

                re_mean, re_std, reprojected_image_points = get_reprojection_error(image_points, world_points,
                                                                                   self.taylor_coefficient, extrinsics,
                                                                                   self.distortion_center,
                                                                                   self.stretch_matrix)

                plt.figure(figsize=(20, 20))
                plt.imshow(im[:, :, [2, 1, 0]])
                plt.scatter(image_points[:, 0], image_points[:, 1], marker="+", c="g", label="detected points")
                plt.scatter(reprojected_image_points[:, 0], reprojected_image_points[:, 1], marker="x", c="r",
                            label="reprojected points")
                plt.scatter(self.distortion_center[0], self.distortion_center[1], c='m', s=20,
                            label="distortion_center")
                plt.scatter(im_center[0], im_center[1], c='c', s=20, label="image center")
                plt.title(
                    f"Linear estimate solution (Reprojection error $ \mu $ = {re_mean:0.2f} $\sigma$ = {re_std:0.2f}). "
                    f"Distortion center = ({self.distortion_center[0]:0.2f}, {self.distortion_center[1]:0.2f})")
                plt.legend()
                plt.show()
                counter += 1

    def show_model_projection(self):
        """
        Get the projection model mapping, i.e. the radius/theta curve with radius the distance from a pixel to the
        distortion center and theta the incidence angle of ray (.wrt z axis).
        :return: radius = f(theta)
        """

        w, h = self.sensor_size
        u = np.arange(0, w, 20).astype(float)
        v = np.arange(0, h, 20).astype(float)
        u, v = np.meshgrid(u, v)
        uv_points = np.vstack((u.flatten(), v.flatten())).T

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

        theta = np.arctan2(np.sqrt(world_points[:, 0] ** 2 + world_points[:, 1] ** 2), world_points[:, 2])

        theta = np.degrees(theta)
        r_calibrated = rho / np.max(rho)

        r_rect, theta_rect = get_canonical_projection_model("rectilinear", 240)
        r_equisolid, theta_equisolid = get_canonical_projection_model("equisolid", 240)
        r_equidistant, theta_equidistant = get_canonical_projection_model("equidistant", 240)
        r_stereographic, theta_stereographic = get_canonical_projection_model("stereographic", 240)

        plt.figure(figsize=(20, 20))
        plt.plot(theta, r_calibrated, c='r', label=" calibrated camera")
        plt.plot(theta_rect, r_rect, c='b', label="rectilinear")
        plt.plot(theta_equisolid, r_equisolid, c='m', label="equisolid")
        plt.plot(theta_equidistant, r_equidistant, c='k', label="equidistant")
        plt.plot(theta_stereographic, r_stereographic, c='b', label="stereographic")
        plt.xlabel("Incident angle in degree", fontsize=15)
        plt.ylabel("Radius / focal_length", fontsize=15)
        plt.title(f"Projection model of {self.cam_name}", fontsize=20)
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(f"./../../../docs/Model_projection_{self.cam_name}.png", dpi=300)
        plt.show()

        return r_calibrated, theta

    def save_calibration(self):
        """
        Save calibration results in .json file
        :return: None
        """
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        outputs = {"date": dt_string,
                   "camera_name": self.cam_name,
                   "valid": self.valid_pattern,
                   "taylor_coefficient": self.taylor_coefficient.tolist(),
                   "distortion_center": self.distortion_center,
                   "stretch_matrix": self.stretch_matrix.tolist(),
                   "inverse_poly": self.inverse_poly.tolist(),
                   "extrinsics_t": [e.tolist() for e in self.extrinsics_t],
                   "img_path": self.images_path,
                   "rms_overall": self.rms_overall,
                   "rms_mean_list": self.rms_mean_list,
                   "rms_std_list": self.rms_std_list
                   }

        with open(f'./../checkpoints/calibration/calibration_{self.cam_name}_{dt_string}.json', 'w') as f:
            json.dump(outputs, f, indent=4)

    def find_poly_inv(self,
                      nb_sample: int = 100,
                      sample_ratio: float = 0.9,
                      max_degree_inverse_poly: int = 25
                      ):
        """
              Find an approximation of the inverse function. New function is much faster !
              :return:
              """
        if self.taylor_coefficient is None or self.distortion_center is None:
            raise ValueError("Fisheye parameters are empty. You first need to specify or load camera's parameters.")

        if sample_ratio < 0 or sample_ratio > 1:
            raise ValueError(f"sample_ratio have to be between 0 and 1. sample_ratio={sample_ratio} is not allow.")

        logger.info("Start searching approximation of the inverse function...")

        theta = np.linspace(0, np.pi * sample_ratio, nb_sample)
        rho = []
        for i in range(nb_sample):
            taylor_tmp = self.taylor_coefficient[::-1].copy()
            taylor_tmp[-2] -= np.tan(np.pi / 2 - theta[i])
            roots = np.roots(taylor_tmp)
            roots = roots[(roots > 0) & (np.imag(roots) == 0)]
            roots = np.array([float(np.real(e)) for e in roots])
            if roots.shape[0] == 0:
                rho.append(np.nan)
            else:
                rho.append(np.min(roots))

        rho = np.array(rho)
        max_error = float("inf")
        deg = 1

        # Repeat until the reprojection error is smaller than 0.01 pixels
        while (max_error > 0.01) & (deg < max_degree_inverse_poly):
            inv_coefficient = np.polyfit(theta, rho, deg)
            rho_inv = np.polyval(inv_coefficient, theta)
            max_error = np.max(np.abs(rho - rho_inv))
            deg += 1
        import matplotlib.pyplot as plt
        logger.info("Poly fit end with success.")
        logger.info(f"Reprojection Error : {max_error:0.4f}")
        logger.info(f"Reprojection polynomial degree: {deg}")
        logger.info(f"Inverse coefficients : {inv_coefficient}")
        self.inverse_poly = inv_coefficient
