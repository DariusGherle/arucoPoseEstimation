"""
Structuri de date pentru sistemul de detectare ArUco 3D
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ArUcoDetection:
    """Structura de date pentru o detectie ArUco"""
    marker_id: int
    corners: List[List[float]]  # 4 colturi ale markerului
    center: Tuple[float, float]  # centrul markerului in pixeli
    rvec: List[float]  # vectorul de rotatie (Rodriguez)
    tvec: List[float]  # vectorul de translatie
    distance: float  # distanta de la camera
    timestamp: float  # timestamp-ul detectiei
    camera_id: str  # ID-ul camerei care a detectat
    camera_position: List[float]  # pozitia camerei in spatiul 3D
    camera_orientation: List[float]  # orientarea camerei


@dataclass
class CameraConfig:
    """Configuratia unei camere"""
    camera_id: str
    position: List[float]  # [x, y, z] in metri
    orientation: List[float]  # [roll, pitch, yaw] in grade


@dataclass
class NetworkMessage:
    """Mesaj pentru comunicarea intre camere"""
    camera_id: str
    detections: List[dict]  # Lista de detectii serializate
    timestamp: float
    message_type: str = "detection_update"