import cv2
import numpy as np
import socket
import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


class ArUcoDetector:
    """Clasa principala pentru detectarea ArUco si calculul pozitiei 3D"""

    def __init__(self, camera_id: str, camera_position: List[float] = [0, 0, 0]):
        self.camera_id = camera_id
        self.camera_position = np.array(camera_position)
        self.camera_orientation = np.array([0, 0, 0])  # roll, pitch, yaw

        # Parametrii camerei (trebuie calibrata pentru rezultate precise)
        # Acesti parametri vor fi ajustati automat pe baza rezolutiei
        self.camera_matrix = np.array([
            [800, 0, 320],  # se va actualiza automat
            [0, 800, 240],  # se va actualiza automat
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Dimensiunea reala a markerului in metri
        self.marker_length = 0.05  # 5 cm

        # Lista de detectii locale
        self.local_detections: List[ArUcoDetection] = []

        # Detectii de la alte camere (primite prin retea)
        self.remote_detections: Dict[str, List[ArUcoDetection]] = {}

        # Network setup
        self.server_port = 8888
        self.client_connections: List[str] = []
        self.server_socket = None
        self.running = False

    def calibrate_camera_from_chessboard(self, chessboard_images_folder: str):
        """Calibreaza camera folosind imagini cu tabla de sah"""
        # Dimensiunile tablei de sah (numarul de colturi interioare)
        chessboard_size = (9, 6)

        # Pregateste punctele obiect (coordonatele 3D ale colturilor tablei)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Arrays pentru a stoca punctele obiect si punctele imagine
        objpoints = []  # puncte 3D in spatiul real
        imgpoints = []  # puncte 2D in planul imaginii

        import glob
        images = glob.glob(f'{chessboard_images_folder}/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Gaseste colturile tablei de sah
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) > 0:
            # Calibreaza camera
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
        # Estimeaza focal length pe baza rezolutiei
        # Regula generala: focal_length ≈ width pentru FOV normal (~60°)
        focal_length = width * 0.8  # Ajusteaza pentru FOV mai larg

        # Centrul imaginii
        cx = width / 2.0
        cy = height / 2.0

        # Actualizeaza matricea
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def detect_aruco_markers(self, frame: np.ndarray) -> List[ArUcoDetection]:
        """Detecteaza markerii ArUco in cadru si calculeaza pozitia 3D"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecteaza markerii
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        current_time = time.time()

        if ids is not None:
            # Estimeaza pozitia pentru fiecare marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                # Calculeaza centrul markerului
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])

                # Vectorii de rotatie si translatie
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Calculeaza distanta (norma vectorului de translatie)
                distance = np.linalg.norm(tvec)

                # Creeaza detectia
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

    def calculate_3d_vector_from_detection(self, detection: ArUcoDetection) -> np.ndarray:
        """Calculeaza vectorul 3D de la camera la marker"""
        # Converteste vectorul de translatie din coordonate camera la coordonate glob ale
        tvec = np.array(detection.tvec)

        # Aplica orientarea camerei (daca exista)
        camera_rotation = R.from_euler('xyz', detection.camera_orientation)
        global_tvec = camera_rotation.apply(tvec)

        # Adauga pozitia camerei pentru coordonate globale
        global_position = np.array(detection.camera_position) + global_tvec

        return global_position

    def calculate_angle_with_camera_axis(self, detection: ArUcoDetection) -> Dict[str, float]:
        """Calculeaza unghiurile markerului fata de axele camerei"""
        rvec = np.array(detection.rvec)
        tvec = np.array(detection.tvec)

        # Converteste vectorul de rotatie la matrice de rotatie
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Converteste la unghi uri Euler (roll, pitch, yaw)
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=True)

        # Calculeaza unghiul dintre directia la marker si axa Z a camerei
        direction_to_marker = tvec / np.linalg.norm(tvec)
        camera_z_axis = np.array([0, 0, 1])
        angle_with_z = np.arccos(np.clip(np.dot(direction_to_marker, camera_z_axis), -1, 1))
        angle_with_z_deg = np.degrees(angle_with_z)

        # Calculeaza unghiul azimut (in planul XY)
        azimuth = np.degrees(np.arctan2(tvec[1], tvec[0]))

        # Calculeaza unghiul de elevatie
        elevation = np.degrees(np.arctan2(tvec[2], np.sqrt(tvec[0] ** 2 + tvec[1] ** 2)))

        return {
            'roll': float(euler_angles[0]),
            'pitch': float(euler_angles[1]),
            'yaw': float(euler_angles[2]),
            'angle_with_camera_z': float(angle_with_z_deg),
            'azimuth': float(azimuth),
            'elevation': float(elevation)
        }

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
        # Rezolva sistemul pentru punctul de intersectie al celor doua linii
        w = cam1_pos - cam2_pos
        u = global_dir1
        v = global_dir2

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
        point1 = cam1_pos + t1 * global_dir1
        point2 = cam2_pos + t2 * global_dir2

        # Pozitia 3D estimata este mijlocul celor doua puncte
        triangulated_position = (point1 + point2) / 2

        return triangulated_position

    def start_network_server(self):
        """Porneste serverul pentru a primi date de la alte camere"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.server_port))
        self.server_socket.listen(5)

        print(f"Server pornit pe portul {self.server_port}")

        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Conexiune noua de la {addr}")

                # Creaza thread pentru a gestiona clientul
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()

            except socket.error:
                break

    def handle_client(self, client_socket: socket.socket, addr):
        """Gestioneaza datele primite de la un client"""
        try:
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break

                try:
                    # Decodifica datele JSON
                    message = json.loads(data.decode('utf-8'))

                    if 'detections' in message:
                        camera_id = message['camera_id']
                        detections_data = message['detections']

                        # Converteste datele inapoi la obiecte ArUcoDetection
                        detections = []
                        for det_data in detections_data:
                            detection = ArUcoDetection(**det_data)
                            detections.append(detection)

                        # Stocheaza detectiile remote
                        self.remote_detections[camera_id] = detections

                        print(f"Primit {len(detections)} detectii de la {camera_id}")

                except json.JSONDecodeError:
                    print(f"Eroare la decodarea JSON de la {addr}")

        except socket.error:
            pass
        finally:
            client_socket.close()
            print(f"Conexiune inchisa cu {addr}")

    def send_detections_to_peer(self, peer_ip: str, peer_port: int, detections: List[ArUcoDetection]):
        """Trimite detectiile la o alta camera"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((peer_ip, peer_port))

            # Pregateste mesajul
            message = {
                'camera_id': self.camera_id,
                'detections': [asdict(det) for det in detections]
            }

            # Trimite datele
            client_socket.send(json.dumps(message).encode('utf-8'))
            client_socket.close()

        except socket.error as e:
            print(f"Eroare la trimiterea datelor catre {peer_ip}:{peer_port} - {e}")

    def get_synchronized_detections(self, time_window: float = 0.1) -> Dict[int, List[ArUcoDetection]]:
        """Returneaza detectiile sincronizate pe markeri in fereastra de timp data"""
        current_time = time.time()
        synchronized = {}

        # Combina detectiile locale si remote
        all_detections = []

        # Adauga detectiile locale recente
        for det in self.local_detections:
            if current_time - det.timestamp <= time_window:
                all_detections.append(det)

        # Adauga detectiile remote recente
        for camera_id, detections in self.remote_detections.items():
            for det in detections:
                if current_time - det.timestamp <= time_window:
                    all_detections.append(det)

        # Grupeaza pe marker_id
        for det in all_detections:
            if det.marker_id not in synchronized:
                synchronized[det.marker_id] = []
            synchronized[det.marker_id].append(det)

        return synchronized

    def visualize_3d_positions(self, triangulated_positions: Dict[int, np.ndarray]):
        """Vizualizeaza pozitiile 3D ale markerilor"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Ploteza camerele
        all_cameras = set()
        for detections in self.remote_detections.values():
            for det in detections:
                all_cameras.add((det.camera_id, tuple(det.camera_position)))

        # Adauga camera locala
        all_cameras.add((self.camera_id, tuple(self.camera_position)))

        # Ploteza pozitiile camerelor
        for camera_id, pos in all_cameras:
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='^', label=f'Camera {camera_id}')

        # Ploteza pozitiile triangulate ale markerilor
        for marker_id, pos in triangulated_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], c='blue', s=150, marker='o', label=f'Marker {marker_id}')
            ax.text(pos[0], pos[1], pos[2], f'  M{marker_id}', fontsize=10)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Pozitii 3D - Camere si Markeri ArUco')
        ax.legend()

        plt.show()

    def start(self):
        """Porneste sistemul complet"""
        self.running = True

        # Porneste serverul de retea
        server_thread = threading.Thread(target=self.start_network_server)
        server_thread.daemon = True
        server_thread.start()

        # Porneste captura video
        cap = cv2.VideoCapture(0)

        # Seteaza rezolutia maxima suportata de camera
        # Incearca sa seteze rezolutia la 1920x1080 (Full HD)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Verifica ce rezolutie s-a setat efectiv
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Rezolutie camera: {actual_width}x{actual_height} @ {fps:.1f} FPS")
        print("Sistem pornit. Apasa 'q' pentru a iesi.")
        print("Apasa 'v' pentru a vizualiza pozitiile 3D")
        print("Apasa 'c' pentru a afisa detectiile curente")
        print("Apasa 'f' pentru a comuta intre fereastra normala si fullscreen")
        print("Apasa 'r' pentru a afisa informatii despre rezolutie")

        try:
            fullscreen = False
            window_name = f'ArUco Detection - {self.camera_id}'

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Actualizeaza parametrii camerei pe baza rezolutiei reale
                height, width = frame.shape[:2]
                self.update_camera_matrix(width, height)

                # Detecteaza markerii
                detections = self.detect_aruco_markers(frame)

                # Actualizeaza detectiile locale
                self.local_detections = detections

                # Deseneaza detectiile pe cadru
                display_frame = frame.copy()

                # Adauga informatii despre FOV in coltul din stanga sus
                fov_info = f"FOV: {width}x{height} | Detectii: {len(detections)}"
                cv2.putText(display_frame, fov_info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                for detection in detections:
                    # Deseneaza conturul markerului
                    corners = np.array(detection.corners, dtype=np.int32)
                    cv2.polylines(display_frame, [corners], True, (0, 255, 0), 3)

                    # Deseneaza centrul
                    center = (int(detection.center[0]), int(detection.center[1]))
                    cv2.circle(display_frame, center, 8, (255, 0, 0), -1)

                    # Deseneaza axa coordonatelor (optinal)
                    rvec = np.array(detection.rvec)
                    tvec = np.array(detection.tvec)
                    cv2.drawFrameAxes(display_frame, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, 0.03)

                    # Calculeaza unghiurile
                    angles = self.calculate_angle_with_camera_axis(detection)

                    # Determina pozitia pentru text (deasupra markerului)
                    text_x = center[0] - 80
                    text_y = center[1] - 60

                    # Afiseaza informatiile cu text mult mai mare si negru cu contur alb
                    # Linia 1: ID si Distanta
                    info_text = f"ID:{detection.marker_id} D:{detection.distance:.2f}m"

                    # Contur alb pentru text negru
                    cv2.putText(display_frame, info_text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)  # Contur alb gros
                    cv2.putText(display_frame, info_text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # Text negru

                    # Linia 2: Azimut si Elevatie
                    angle_text = f"Az:{angles['azimuth']:.1f}° El:{angles['elevation']:.1f}°"

                    # Contur alb pentru text negru
                    cv2.putText(display_frame, angle_text,
                                (text_x, text_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5)  # Contur alb
                    cv2.putText(display_frame, angle_text,
                                (text_x, text_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)  # Text negru

                    # Linia 3: Pozitie 3D
                    pos_3d = self.calculate_3d_vector_from_detection(detection)
                    pos_text = f"3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})"

                    # Contur alb pentru text negru
                    cv2.putText(display_frame, pos_text,
                                (text_x, text_y + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)  # Contur alb
                    cv2.putText(display_frame, pos_text,
                                (text_x, text_y + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Text negru

                # Afiseaza fereastra
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Comuta intre fullscreen si fereastra normala
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("Mod fullscreen activat")
                    else:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("Mod fereastra normala activat")
                elif key == ord('r'):
                    # Afiseaza informatii despre rezolutie si FOV
                    print(f"\n=== Informatii Camera ===")
                    print(f"Rezolutie: {width}x{height}")
                    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
                    print(f"FOV effective: 100% vizibil")
                    print(f"Camera matrix:")
                    print(self.camera_matrix)
                elif key == ord('v'):
                    # Vizualizeaza pozitiile 3D
                    synced_detections = self.get_synchronized_detections()
                    triangulated = {}

                    for marker_id, dets in synced_detections.items():
                        if len(dets) >= 2:
                            pos = self.triangulate_3d_position(dets)
                            if pos is not None:
                                triangulated[marker_id] = pos

                    if triangulated:
                        self.visualize_3d_positions(triangulated)
                    else:
                        print("Nu exista suficiente detectii pentru triangulatie")

                elif key == ord('c'):
                    # Afiseaza detectiile curente
                    print(f"\n=== Detectii curente - {self.camera_id} ===")
                    for det in detections:
                        angles = self.calculate_angle_with_camera_axis(det)
                        print(f"Marker {det.marker_id}: Distanta={det.distance:.2f}m, "
                              f"Azimut={angles['azimuth']:.1f}°, Elevatie={angles['elevation']:.1f}°")

                    synced = self.get_synchronized_detections()
                    print(f"Detectii sincronizate de la toate camerele: {len(synced)} markeri")
                    for marker_id, dets in synced.items():
                        print(f"  Marker {marker_id}: {len(dets)} detectii din camere diferite")

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            if self.server_socket:
                self.server_socket.close()


def main():
    """Functia principala - exemplu de utilizare"""
    print("=== Sistem de Detectare ArUco si Triangulatie 3D Distribuita ===")
    print()

    # Citeste configuratia
    camera_id = input("Introduceti ID-ul acestei camere (ex: 'cam1'): ").strip() or 'cam1'

    # Pozitia camerei in spatiul 3D (coordonate globale)
    print("Introduceti pozitia camerei in spatiul 3D (metri):")
    x = float(input("X: ") or "0")
    y = float(input("Y: ") or "0")
    z = float(input("Z: ") or "0")

    # Creeaza detectorul
    detector = ArUcoDetector(camera_id, [x, y, z])

    # Optinal: Adauga camere peer pentru sincronizare
    add_peers = input("Doriti sa adaugati camere peer? (y/n): ").strip().lower()
    peers = []

    if add_peers == 'y':
        while True:
            peer_ip = input("IP peer (sau Enter pentru a termina): ").strip()
            if not peer_ip:
                break
            peer_port = int(input(f"Port peer pentru {peer_ip} (default 8888): ") or "8888")
            peers.append((peer_ip, peer_port))

    print("\nConfiguratie:")
    print(f"Camera ID: {camera_id}")
    print(f"Pozitie: ({x}, {y}, {z})")
    print(f"Peers: {peers}")
    print(f"Server port: {detector.server_port}")
    print()

    # Porneste sistemul
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\nSistem oprit de utilizator")


if __name__ == "__main__":
    main()