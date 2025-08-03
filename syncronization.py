"""
Modulul pentru sincronizarea detectiilor intre camere
"""

import time
from typing import List, Dict, Optional
from collections import defaultdict

from data_structures import ArUcoDetection


class SynchronizationManager:
    """Clasa pentru managementul sincronizarii detectiilor"""

    def __init__(self, time_window: float = 0.1):
        self.time_window = time_window  # Fereastra de sincronizare in secunde

        # Detectii locale si remote
        self.local_detections: List[ArUcoDetection] = []
        self.remote_detections: Dict[str, List[ArUcoDetection]] = {}

        # Istoric pentru cleanup
        self.detection_history: List[ArUcoDetection] = []
        self.max_history_size = 1000

    def update_local_detections(self, detections: List[ArUcoDetection]):
        """Actualizeaza detectiile locale"""
        self.local_detections = detections

        # Adauga la istoric
        self.detection_history.extend(detections)

        # Cleanup istoric
        self._cleanup_history()

    def update_remote_detections(self, camera_id: str, detections: List[ArUcoDetection]):
        """Actualizeaza detectiile de la o camera remote"""
        self.remote_detections[camera_id] = detections

        # Adauga la istoric
        self.detection_history.extend(detections)

        # Cleanup istoric
        self._cleanup_history()

    def get_synchronized_detections(self, reference_time: Optional[float] = None) -> Dict[int, List[ArUcoDetection]]:
        """Returneaza detectiile sincronizate pe markeri"""
        if reference_time is None:
            reference_time = time.time()

        synchronized = defaultdict(list)

        # Colecteaza toate detectiile din fereastra de timp
        all_detections = []

        # Detectii locale
        for det in self.local_detections:
            if abs(reference_time - det.timestamp) <= self.time_window:
                all_detections.append(det)

        # Detectii remote
        for camera_id, detections in self.remote_detections.items():
            for det in detections:
                if abs(reference_time - det.timestamp) <= self.time_window:
                    all_detections.append(det)

        # Grupeaza pe marker_id
        for det in all_detections:
            synchronized[det.marker_id].append(det)

        return dict(synchronized)

    def get_best_synchronized_detections(self) -> Dict[int, List[ArUcoDetection]]:
        """Returneaza cea mai buna sincronizare din fereastra curenta"""
        current_time = time.time()

        # Incearca diferite momente de referinta in fereastra
        best_sync = {}
        max_total_detections = 0

        # Testeaza diferite puncte in fereastra de timp
        for offset in [-self.time_window / 2, 0, self.time_window / 2]:
            ref_time = current_time + offset
            sync = self.get_synchronized_detections(ref_time)

            # Calculeaza scorul total (mai multe detectii = mai bun)
            total_detections = sum(len(dets) for dets in sync.values())

            if total_detections > max_total_detections:
                max_total_detections = total_detections
                best_sync = sync

        return best_sync

    def get_detection_statistics(self) -> Dict[str, any]:
        """Returneaza statistici despre detectii"""
        current_time = time.time()

        # Detectii recente (ultima secunda)
        recent_local = len([d for d in self.local_detections
                            if current_time - d.timestamp <= 1.0])

        recent_remote = {}
        total_remote = 0
        for camera_id, detections in self.remote_detections.items():
            recent = len([d for d in detections
                          if current_time - d.timestamp <= 1.0])
            recent_remote[camera_id] = recent
            total_remote += recent

        # Markeri unici detectati
        synced = self.get_synchronized_detections()
        unique_markers = len(synced)
        triangulatable_markers = len([mid for mid, dets in synced.items()
                                      if len(dets) >= 2])

        return {
            'local_detections_recent': recent_local,
            'remote_detections_recent': recent_remote,
            'total_remote_recent': total_remote,
            'unique_markers': unique_markers,
            'triangulatable_markers': triangulatable_markers,
            'connected_cameras': len(self.remote_detections),
            'time_window': self.time_window,
            'history_size': len(self.detection_history)
        }

    def _cleanup_history(self):
        """Curata istoricul de detectii"""
        if len(self.detection_history) > self.max_history_size:
            # Pastreaza doar detectiile recente
            current_time = time.time()
            self.detection_history = [
                det for det in self.detection_history
                if current_time - det.timestamp <= 5.0  # Ultimele 5 secunde
            ]

    def set_time_window(self, window: float):
        """Seteaza fereastra de sincronizare"""
        self.time_window = max(0.01, min(1.0, window))  # Intre 10ms si 1s

    def get_camera_sync_quality(self, camera_id: str) -> Dict[str, float]:
        """Evalueaza calitatea sincronizarii cu o camera"""
        if camera_id not in self.remote_detections:
            return {'quality': 0.0, 'latency': float('inf'), 'detection_rate': 0.0}

        current_time = time.time()
        remote_dets = self.remote_detections[camera_id]

        if not remote_dets:
            return {'quality': 0.0, 'latency': float('inf'), 'detection_rate': 0.0}

        # Calculeaza latenta medie
        latencies = [current_time - det.timestamp for det in remote_dets]
        avg_latency = sum(latencies) / len(latencies)

        # Rata de detectie (detectii/secunda)
        recent_dets = [det for det in remote_dets
                       if current_time - det.timestamp <= 1.0]
        detection_rate = len(recent_dets)

        # Scor de calitate (invers proportional cu latenta)
        quality = max(0.0, 1.0 - avg_latency / self.time_window)

        return {
            'quality': quality,
            'latency': avg_latency,
            'detection_rate': detection_rate
        }

    def get_all_cameras_sync_quality(self) -> Dict[str, Dict[str, float]]:
        """Returneaza calitatea sincronizarii pentru toate camerele"""
        result = {}
        for camera_id in self.remote_detections.keys():
            result[camera_id] = self.get_camera_sync_quality(camera_id)
        return result

    def clear_old_detections(self, max_age: float = 2.0):
        """Sterge detectiile mai vechi decat max_age secunde"""
        current_time = time.time()

        # Curata detectiile locale
        self.local_detections = [
            det for det in self.local_detections
            if current_time - det.timestamp <= max_age
        ]

        # Curata detectiile remote
        for camera_id in list(self.remote_detections.keys()):
            self.remote_detections[camera_id] = [
                det for det in self.remote_detections[camera_id]
                if current_time - det.timestamp <= max_age
            ]

            # Sterge camerele care nu au trimis detectii recente
            if not self.remote_detections[camera_id]:
                del self.remote_detections[camera_id]

    def get_marker_tracking_info(self, marker_id: int) -> Dict[str, any]:
        """Returneaza informatii de tracking pentru un marker specific"""
        current_time = time.time()

        # Gaseste toate detectiile pentru acest marker
        marker_detections = []

        # Locale
        for det in self.local_detections:
            if det.marker_id == marker_id:
                marker_detections.append(det)

        # Remote
        for camera_id, detections in self.remote_detections.items():
            for det in detections:
                if det.marker_id == marker_id:
                    marker_detections.append(det)

        if not marker_detections:
            return {'visible': False}

        # Camerele care vad markerul
        cameras_seeing = set(det.camera_id for det in marker_detections)

        # Ultima detectie
        latest_detection = max(marker_detections, key=lambda d: d.timestamp)
        last_seen = current_time - latest_detection.timestamp

        # Stabilitatea (varianta pozitiei)
        if len(marker_detections) > 1:
            positions = [det.tvec for det in marker_detections[-5:]]  # Ultimele 5
            avg_pos = [sum(p[i] for p in positions) / len(positions) for i in range(3)]
            variance = sum(
                sum((pos[i] - avg_pos[i]) ** 2 for i in range(3))
                for pos in positions
            ) / len(positions)
            stability = max(0.0, 1.0 - variance)
        else:
            stability = 1.0

        return {
            'visible': True,
            'marker_id': marker_id,
            'cameras_seeing': list(cameras_seeing),
            'camera_count': len(cameras_seeing),
            'last_seen_seconds': last_seen,
            'stability': stability,
            'triangulatable': len(cameras_seeing) >= 2,
            'total_detections': len(marker_detections)
        }