"""
Modulul pentru vizualizarea 3D si interfata utilizator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional

from data_structures import ArUcoDetection


class Visualizer3D:
    """Clasa pentru vizualizarea 3D a sistemului"""

    def __init__(self):
        self.fig = None
        self.ax = None

    def visualize_3d_positions(self, triangulated_positions: Dict[int, np.ndarray],
                               camera_positions: Dict[str, np.ndarray],
                               detections: Dict[int, List[ArUcoDetection]] = None):
        """Vizualizeaza pozitiile 3D ale markerilor si camerelor"""

        # Creaza figura 3D
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Ploteza camerele
        for camera_id, pos in camera_positions.items():
            self.ax.scatter(pos[0], pos[1], pos[2],
                            c='red', s=150, marker='^',
                            label=f'Camera {camera_id}')

            # Adauga text pentru camera
            self.ax.text(pos[0], pos[1], pos[2] + 0.1,
                         f'  {camera_id}', fontsize=10, color='red')

        # Ploteza markerii triangulati
        for marker_id, pos in triangulated_positions.items():
            # Culoare diferita pentru fiecare marker
            colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
            color = colors[marker_id % len(colors)]

            self.ax.scatter(pos[0], pos[1], pos[2],
                            c=color, s=200, marker='o',
                            alpha=0.8, edgecolors='black', linewidth=2)

            # Adauga text pentru marker
            self.ax.text(pos[0], pos[1], pos[2] + 0.05,
                         f'M{marker_id}', fontsize=12, weight='bold')

        # Desenaza linii de la camere la markeri (daca sunt detectii)
        if detections:
            for marker_id, det_list in detections.items():
                if marker_id in triangulated_positions:
                    marker_pos = triangulated_positions[marker_id]

                    for detection in det_list:
                        cam_pos = np.array(detection.camera_position)

                        # Linie punctata de la camera la marker
                        self.ax.plot([cam_pos[0], marker_pos[0]],
                                     [cam_pos[1], marker_pos[1]],
                                     [cam_pos[2], marker_pos[2]],
                                     'k--', alpha=0.3, linewidth=1)

        # Configurare axe
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.set_title('Pozitii 3D - Camere si Markeri ArUco', fontsize=14, weight='bold')

        # Legende
        self.ax.legend(loc='upper right')

        # Grid si stil
        self.ax.grid(True, alpha=0.3)

        # Seteaza aspectul egal pentru axe
        max_range = 2.0  # metri
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range])

        plt.tight_layout()
        plt.show()

    def create_real_time_plot(self):
        """Creaza un plot pentru actualizare in timp real"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 8))

    def update_real_time_plot(self, triangulated_positions: Dict[int, np.ndarray],
                              camera_positions: Dict[str, np.ndarray]):
        """Actualizeaza plot-ul in timp real"""
        if self.ax is None:
            self.create_real_time_plot()

        self.ax.clear()

        # Redeseneaza tot
        for camera_id, pos in camera_positions.items():
            self.ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='^')
            self.ax.text(pos[0], pos[1], pos[2], f'  {camera_id}', fontsize=8)

        for marker_id, pos in triangulated_positions.items():
            self.ax.scatter(pos[0], pos[1], pos[2], c='blue', s=100, marker='o')
            self.ax.text(pos[0], pos[1], pos[2], f'  M{marker_id}', fontsize=8)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        plt.draw()
        plt.pause(0.01)


class FrameRenderer:
    """Clasa pentru renderizarea frame-urilor video cu detectii"""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_detections_on_frame(self, frame: np.ndarray,
                                   detections: List[ArUcoDetection],
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray,
                                   show_axes: bool = True,
                                   show_3d_info: bool = True) -> np.ndarray:
        """Rendereaza detectiile pe frame"""

        display_frame = frame.copy()
        height, width = frame.shape[:2]

        # Header cu informatii generale
        header_text = f"FOV: {width}x{height} | Detectii: {len(detections)}"
        cv2.putText(display_frame, header_text, (10, 30),
                    self.font, 0.7, (0, 255, 255), 2)

        for detection in detections:
            self._render_single_detection(display_frame, detection,
                                          camera_matrix, dist_coeffs,
                                          show_axes, show_3d_info)

        return display_frame

    def _render_single_detection(self, frame: np.ndarray,
                                 detection: ArUcoDetection,
                                 camera_matrix: np.ndarray,
                                 dist_coeffs: np.ndarray,
                                 show_axes: bool,
                                 show_3d_info: bool):
        """Rendereaza o singura detectie"""

        # Conturul markerului
        corners = np.array(detection.corners, dtype=np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 3)

        # Centrul
        center = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(frame, center, 8, (255, 0, 0), -1)

        # Axele coordinate
        if show_axes:
            rvec = np.array(detection.rvec)
            tvec = np.array(detection.tvec)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec, tvec, 0.03)

        # Text cu informatii
        if show_3d_info:
            self._render_detection_text(frame, detection, center)

    def _render_detection_text(self, frame: np.ndarray,
                               detection: ArUcoDetection,
                               center: tuple):
        """Rendereaza textul cu informatii despre detectie"""

        # Calculeaza unghiurile (simplified version)
        tvec = np.array(detection.tvec)
        azimuth = np.degrees(np.arctan2(tvec[1], tvec[0]))
        elevation = np.degrees(np.arctan2(tvec[2], np.sqrt(tvec[0] ** 2 + tvec[1] ** 2)))

        # Pozitionarea textului
        text_x = center[0] - 80
        text_y = center[1] - 60

        # Linia 1: ID si Distanta
        info_text = f"ID:{detection.marker_id} D:{detection.distance:.2f}m"
        self._draw_outlined_text(frame, info_text, (text_x, text_y), 1.2)

        # Linia 2: Azimut si Elevatie
        angle_text = f"Az:{azimuth:.1f}° El:{elevation:.1f}°"
        self._draw_outlined_text(frame, angle_text, (text_x, text_y + 40), 1.0)

        # Linia 3: Pozitie 3D
        pos_3d = detection.camera_position
        pos_text = f"3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})"
        self._draw_outlined_text(frame, pos_text, (text_x, text_y + 80), 0.8)

    def _draw_outlined_text(self, frame: np.ndarray, text: str,
                            position: tuple, scale: float):
        """Deseneaza text cu contur alb si interior negru"""
        thickness_outline = max(1, int(scale * 4))
        thickness_text = max(1, int(scale * 2))

        # Contur alb
        cv2.putText(frame, text, position, self.font, scale,
                    (255, 255, 255), thickness_outline)

        # Text negru
        cv2.putText(frame, text, position, self.font, scale,
                    (0, 0, 0), thickness_text)

    def create_info_overlay(self, frame: np.ndarray,
                            stats: Dict[str, any],
                            sync_quality: Dict[str, Dict[str, float]] = None) -> np.ndarray:
        """Creaza un overlay cu informatii despre sistem"""

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Background semi-transparent pentru informatii
        overlay_bg = np.zeros((200, 300, 3), dtype=np.uint8)
        overlay_bg[:] = (50, 50, 50)  # Gri inchis

        # Informatii de baza
        y_offset = 30
        info_texts = [
            f"Detectii locale: {stats.get('local_detections_recent', 0)}",
            f"Detectii remote: {stats.get('total_remote_recent', 0)}",
            f"Markeri unici: {stats.get('unique_markers', 0)}",
            f"Triangulabili: {stats.get('triangulatable_markers', 0)}",
            f"Camere conectate: {stats.get('connected_cameras', 0)}"
        ]

        for i, text in enumerate(info_texts):
            cv2.putText(overlay_bg, text, (10, y_offset + i * 25),
                        self.font, 0.5, (255, 255, 255), 1)

        # Adauga overlay-ul pe frame
        overlay[10:210, w - 310:w - 10] = overlay_bg

        return overlay