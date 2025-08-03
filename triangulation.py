"""
Modulul pentru triangulația 3D și calculele geometrice
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R

from data_structures import ArUcoDetection


class Triangulator3D:
    """Clasa pentru triangulația 3D a markerilor detectați din mai multe camere"""

    def __init__(self):
        pass

    def triangulate_3d_position(self, detections_same_marker: List[ArUcoDetection]) -> Optional[np.ndarray]:
        """Trianguleaza pozitia 3D folosind detectii de la mai multe camere"""
        if len(detections_same_marker) < 2:
            return None

        # Foloseste primele doua detectii pentru triangulatie simpla
        det1, det2 = detections_same_marker[0], detections_same_marker[1]

        # Pozitiile camerelor in spatiul 3D
        cam1_pos = np.array(det1.camera_position)
        cam2_pos = np.array(det2.camera_position)

        # Directiile de la camere la marker (normalizate)
        dir1 = np.array(det1.tvec) / np.linalg.norm(det1.tvec)
        dir2 = np.array(det2.tvec) / np.linalg.norm(det2.tvec)

        # Aplica orientarea camerelor
        cam1_rot = R.from_euler('xyz', det1.camera_orientation)
        cam2_rot = R.from_euler('xyz', det2.camera_orientation)

        global_dir1 = cam1_rot.apply(dir1)
        global_dir2 = cam2_rot.apply(dir2)

        # Triangulatie prin metoda celui mai apropiat punct
        return self._closest_point_between_lines(cam1_pos, global_dir1, cam2_pos, global_dir2)

    def _closest_point_between_lines(self, p1: np.ndarray, d1: np.ndarray,
                                     p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
        """Gaseste punctul cel mai apropiat dintre doua linii in 3D"""
        w = p1 - p2
        u = d1
        v = d2

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)

        denominator = a * c - b * b
        if abs(denominator) < 1e-6:
            return None  # Liniile sunt paralele

        t1 = (b * e - c * d) / denominator
        t2 = (a * e - b * d) / denominator

        # Punctele cele mai apropiate pe cele doua linii
        point1 = p1 + t1 * u
        point2 = p2 + t2 * v

        # Pozitia 3D estimata
        triangulated_position = (point1 + point2) / 2

        return triangulated_position

    def triangulate_multiple_cameras(self, detections_same_marker: List[ArUcoDetection]) -> Optional[np.ndarray]:
        """Triangulație avansată folosind toate camerele disponibile"""
        if len(detections_same_marker) < 2:
            return None

        if len(detections_same_marker) == 2:
            return self.triangulate_3d_position(detections_same_marker)

        # Pentru 3+ camere, folosește toate combinațiile și calculează media
        triangulated_points = []

        for i in range(len(detections_same_marker)):
            for j in range(i + 1, len(detections_same_marker)):
                point = self.triangulate_3d_position([detections_same_marker[i], detections_same_marker[j]])
                if point is not None:
                    triangulated_points.append(point)

        if not triangulated_points:
            return None

        # Media tuturor punctelor triangulate
        final_position = np.mean(triangulated_points, axis=0)

        return final_position

    def calculate_triangulation_error(self, position_3d: np.ndarray,
                                      detections: List[ArUcoDetection]) -> float:
        """Calculează eroarea de reproiecție pentru validarea triangulației"""
        total_error = 0.0

        for detection in detections:
            # Pozitia camerei și orientarea
            cam_pos = np.array(detection.camera_position)
            cam_rot = R.from_euler('xyz', detection.camera_orientation)

            # Vectorul de la cameră la poziția triangulată
            vector_to_marker = position_3d - cam_pos
            local_vector = cam_rot.inv().apply(vector_to_marker)

            # Vectorul detectat
            detected_vector = np.array(detection.tvec)

            # Eroarea unghiulară
            cos_angle = np.dot(local_vector, detected_vector) / (
                    np.linalg.norm(local_vector) * np.linalg.norm(detected_vector)
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angular_error = np.degrees(np.arccos(cos_angle))

            total_error += angular_error

        return total_error / len(detections)

    def filter_outlier_detections(self, detections: List[ArUcoDetection],
                                  max_angular_error: float = 10.0) -> List[ArUcoDetection]:
        """Filtrează detectările care au erori prea mari"""
        if len(detections) < 3:
            return detections

        # Încearcă triangulația cu toate detectările
        triangulated_pos = self.triangulate_multiple_cameras(detections)
        if triangulated_pos is None:
            return detections

        # Calculează erorile pentru fiecare detectare
        filtered_detections = []
        for detection in detections:
            error = self.calculate_triangulation_error(triangulated_pos, [detection])
            if error <= max_angular_error:
                filtered_detections.append(detection)

        # Returnează detectările filtrate sau originalele dacă sunt prea puține
        return filtered_detections if len(filtered_detections) >= 2 else detections