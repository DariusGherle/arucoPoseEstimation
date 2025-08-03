"""
Modulul principal pentru detectarea markerilor ArUco
"""

import cv2
import numpy as np
import time
import glob
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R

from data_structures import ArUcoDetection


class ArUcoDetector:
    """Clasa pentru detectarea si procesarea markerilor ArUco"""

    def __init__(self, camera_id: str, camera_position: List[float] = [0, 0, 0]):
        self.camera_id = camera_id
        self.camera_position = np.array(camera_position)
        self.camera_orientation = np.array([0, 0, 0])  # roll, pitch, yaw

        # Parametrii camerei (se actualizeaza automat)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Dimensiunea reala a markerului in metri
        self.marker_length = 0.05  # 5 cm

    def calibrate_camera_from_chessboard(self, chessboard_images_folder: str) -> bool:
        """Calibreaza camera folosind imagini cu tabla de sah"""
        chessboard_size = (9, 6)

        # Pregateste punctele obiect
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(f'{chessboard_images_folder}/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                print(f"Camera calibrata cu succes pentru {self.camera_id}")
                return True

        print(f"Eroare la calibrarea camerei {self.camera_id}")
        return False

    def update_camera_matrix(self, width: int, height: int):
        """Actualizeaza matricea camerei pe baza rezolutiei reale"""
        focal_length = width * 0.8
        cx = width / 2.0
        cy = height / 2.0

        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def detect_aruco_markers(self, frame: np.ndarray) -> List[ArUcoDetection]:
        """Detecteaza markerii ArUco in cadru si calculeaza pozitia 3D"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        current_time = time.time()

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])

                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                distance = np.linalg.norm(tvec)

                detection = ArUcoDetection(
                    marker_id=int(marker_id),
                    corners=marker_corners.tolist(),
                    center=(float(center_x), float(center_y)),
                    rvec=rvec.tolist(),
                    tvec=tvec.tolist(),
                    distance=float(distance),
                    timestamp=current_time,
                    camera_id=self.camera_id,
                    camera_position=self.camera_position.tolist(),
                    camera_orientation=self.camera_orientation.tolist()
                )

                detections.append(detection)

        return detections

    def calculate_angle_with_camera_axis(self, detection: ArUcoDetection) -> Dict[str, float]:
        """Calculeaza unghiurile markerului fata de axele camerei"""
        rvec = np.array(detection.rvec)
        tvec = np.array(detection.tvec)

        # Converteste vectorul de rotatie la matrice
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=True)

        # Unghiul cu axa Z a camerei
        direction_to_marker = tvec / np.linalg.norm(tvec)
        camera_z_axis = np.array([0, 0, 1])
        angle_with_z = np.arccos(np.clip(np.dot(direction_to_marker, camera_z_axis), -1, 1))
        angle_with_z_deg = np.degrees(angle_with_z)

        # Azimut si elevatie
        azimuth = np.degrees(np.arctan2(tvec[1], tvec[0]))
        elevation = np.degrees(np.arctan2(tvec[2], np.sqrt(tvec[0] ** 2 + tvec[1] ** 2)))

        return {
            'roll': float(euler_angles[0]),
            'pitch': float(euler_angles[1]),
            'yaw': float(euler_angles[2]),
            'angle_with_camera_z': float(angle_with_z_deg),
            'azimuth': float(azimuth),
            'elevation': float(elevation)
        }

    def calculate_3d_vector_from_detection(self, detection: ArUcoDetection) -> np.ndarray:
        """Calculeaza vectorul 3D de la camera la marker"""
        tvec = np.array(detection.tvec)

        # Aplica orientarea camerei
        camera_rotation = R.from_euler('xyz', detection.camera_orientation)
        global_tvec = camera_rotation.apply(tvec)

        # Coordonate globale
        global_position = np.array(detection.camera_position) + global_tvec

        return global_position