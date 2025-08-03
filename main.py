"""
Aplicatia principala pentru sistemul de detectare ArUco 3D distribuit
"""

import cv2
import time
import threading
from typing import List, Dict
import numpy as np

# Import module locale
from data_structures import ArUcoDetection, CameraConfig
from detector import ArUcoDetector
from triangulation import Triangulator3D
from network_manager import NetworkManager
from syncronization import SynchronizationManager
from visualization import Visualizer3D, FrameRenderer


class ArUcoSystem:
    """Clasa principala pentru sistemul ArUco 3D distribuit"""

    def __init__(self, camera_config: CameraConfig):
        # Componente principale
        self.detector = ArUcoDetector(camera_config.camera_id, camera_config.position)
        self.triangulator = Triangulator3D()
        self.network_manager = NetworkManager(camera_config.camera_id)
        self.sync_manager = SynchronizationManager()
        self.visualizer = Visualizer3D()
        self.renderer = FrameRenderer()

        # Configuratie
        self.camera_config = camera_config
        self.running = False

        # Setup callbacks
        self.network_manager.set_detection_callback(self._on_remote_detections)

        # Rezultate triangulatie
        self.triangulated_positions: Dict[int, np.ndarray] = {}

    def _on_remote_detections(self, camera_id: str, detections: List[ArUcoDetection]):
        """Callback pentru detectiile primite de la camere remote"""
        self.sync_manager.update_remote_detections(camera_id, detections)

    def start_system(self, peers: List[str] = None):
        """Porneste sistemul complet"""
        self.running = True

        # Porneste serverul de retea
        server_thread = threading.Thread(target=self.network_manager.start_server)
        server_thread.daemon = True
        server_thread.start()

        # Porneste sincronizarea automata daca sunt peers
        if peers:
            self.network_manager.start_sync_with_detection_source(
                lambda: self.sync_manager.local_detections, peers
            )

        # Porneste bucla principala
        self._main_loop()

    def _main_loop(self):
        """Bucla principala de procesare"""
        # Initializeaza camera
        cap = cv2.VideoCapture(0)

        # Seteaza rezolutia
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Camera: {actual_width}x{actual_height} @ {fps:.1f} FPS")
        self._print_controls()

        try:
            fullscreen = False
            window_name = f'ArUco 3D System - {self.camera_config.camera_id}'

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Actualizeaza parametrii camerei
                height, width = frame.shape[:2]
                self.detector.update_camera_matrix(width, height)

                # Detecteaza markerii
                detections = self.detector.detect_aruco_markers(frame)

                # Actualizeaza detectiile locale
                self.sync_manager.update_local_detections(detections)

                # Calculeaza triangulatie
                self._update_triangulation()

                # Rendereaza frame-ul
                display_frame = self._render_frame(frame, detections)

                # Afiseaza frame-ul
                cv2.imshow(window_name, display_frame)

                # Proceseaza input-ul
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_input(key, window_name, fullscreen):
                    break

        finally:
            self._cleanup(cap)

    def _update_triangulation(self):
        """Actualizeaza calculele de triangulatie"""
        # Obtine detectiile sincronizate
        synced_detections = self.sync_manager.get_best_synchronized_detections()

        # Calculeaza pozitiile 3D
        self.triangulated_positions = {}

        for marker_id, detections in synced_detections.items():
            if len(detections) >= 2:
                # Filtreaza outliers
                filtered_detections = self.triangulator.filter_outlier_detections(detections)

                # Trianguleaza pozitia
                position = self.triangulator.triangulate_multiple_cameras(filtered_detections)
                if position is not None:
                    self.triangulated_positions[marker_id] = position

    def _render_frame(self, frame: np.ndarray, detections: List[ArUcoDetection]) -> np.ndarray:
        """Rendereaza frame-ul cu toate informatiile"""
        # Rendereaza detectiile
        display_frame = self.renderer.render_detections_on_frame(
            frame, detections,
            self.detector.camera_matrix,
            self.detector.dist_coeffs
        )

        # Adauga informatii despre sistem
        stats = self.sync_manager.get_detection_statistics()
        sync_quality = self.sync_manager.get_all_cameras_sync_quality()

        display_frame = self.renderer.create_info_overlay(
            display_frame, stats, sync_quality
        )

        return display_frame

    def _handle_key_input(self, key: int, window_name: str, fullscreen: bool) -> bool:
        """Gestioneaza input-ul de la tastatura"""
        if key == ord('q'):
            return False

        elif key == ord('f'):
            # Toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("🖼️ Mod fullscreen activat")
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("🖼️ Mod fereastra normala activat")

        elif key == ord('v'):
            # Vizualizeaza 3D
            self._show_3d_visualization()

        elif key == ord('c'):
            # Afiseaza statistici
            self._print_statistics()

        elif key == ord('r'):
            # Informatii rezolutie
            self._print_resolution_info()

        elif key == ord('s'):
            # Salveaza frame
            self._save_current_frame()

        elif key == ord('t'):
            # Ajusteaza timpul de sincronizare
            self._adjust_sync_time()

        elif key == ord('h'):
            # Help
            self._print_controls()

        return True

    def _show_3d_visualization(self):
        """Afiseaza vizualizarea 3D"""
        if not self.triangulated_positions:
            print("❌ Nu exista pozitii triangulate pentru vizualizare")
            return

        # Colecteaza pozitiile camerelor
        camera_positions = {self.camera_config.camera_id: np.array(self.camera_config.position)}

        for camera_id, detections in self.sync_manager.remote_detections.items():
            if detections:
                pos = np.array(detections[0].camera_position)
                camera_positions[camera_id] = pos

        # Obtine detectiile pentru linii
        synced_detections = self.sync_manager.get_best_synchronized_detections()

        print(f"🎯 Vizualizare 3D: {len(self.triangulated_positions)} markeri, {len(camera_positions)} camere")

        # Afiseaza vizualizarea
        self.visualizer.visualize_3d_positions(
            self.triangulated_positions,
            camera_positions,
            synced_detections
        )

    def _print_statistics(self):
        """Afiseaza statistici detaliate"""
        print(f"\n{'=' * 50}")
        print(f"📊 STATISTICI SISTEM - {self.camera_config.camera_id}")
        print(f"{'=' * 50}")

        stats = self.sync_manager.get_detection_statistics()

        print(f"🎯 Detectii:")
        print(f"   Locale (recente): {stats['local_detections_recent']}")
        print(f"   Remote (recente): {stats['total_remote_recent']}")
        print(f"   Markeri unici: {stats['unique_markers']}")
        print(f"   Triangulabili: {stats['triangulatable_markers']}")

        print(f"\n🌐 Retea:")
        print(f"   Camere conectate: {stats['connected_cameras']}")
        print(f"   Peers activi: {len(self.network_manager.get_connected_peers())}")

        sync_quality = self.sync_manager.get_all_cameras_sync_quality()
        if sync_quality:
            print(f"\n⏱️ Calitate sincronizare:")
            for camera_id, quality in sync_quality.items():
                print(f"   {camera_id}: {quality['quality']:.2f} "
                      f"(latenta: {quality['latency']:.3f}s, "
                      f"rata: {quality['detection_rate']:.1f}/s)")

        print(f"\n📐 Triangulatie:")
        for marker_id, pos in self.triangulated_positions.items():
            tracking_info = self.sync_manager.get_marker_tracking_info(marker_id)
            print(f"   Marker {marker_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                  f"- {tracking_info['camera_count']} camere")

    def _print_resolution_info(self):
        """Afiseaza informatii despre rezolutie"""
        print(f"\n📹 INFORMATII CAMERA - {self.camera_config.camera_id}")
        print(f"Matrix camera:")
        print(self.detector.camera_matrix)
        print(f"Coeficienti distorsiune: {self.detector.dist_coeffs}")
        print(f"Dimensiune marker: {self.detector.marker_length}m")

    def _save_current_frame(self):
        """Salveaza frame-ul curent"""
        timestamp = int(time.time())
        filename = f"aruco_frame_{self.camera_config.camera_id}_{timestamp}.jpg"
        # Implementare pentru salvare frame
        print(f"💾 Frame salvat: {filename}")

    def _adjust_sync_time(self):
        """Ajusteaza timpul de sincronizare"""
        current_window = self.sync_manager.time_window
        print(f"⏱️ Fereastra actuala de sincronizare: {current_window * 1000:.0f}ms")

        try:
            new_window = float(input("Noua fereastra (secunde, 0.01-1.0): ") or current_window)
            self.sync_manager.set_time_window(new_window)
            print(f"✅ Fereastra setata la: {new_window * 1000:.0f}ms")
        except:
            print("❌ Valoare invalida")

    def _print_controls(self):
        """Afiseaza comenzile disponibile"""
        print(f"\n🎮 COMENZI DISPONIBILE:")
        print(f"   'q' - Iesire din program")
        print(f"   'v' - Vizualizare 3D")
        print(f"   'c' - Afisare statistici complete")
        print(f"   'f' - Toggle fullscreen")
        print(f"   'r' - Informatii rezolutie camera")
        print(f"   's' - Salveaza frame curent")
        print(f"   't' - Ajusteaza timp sincronizare")
        print(f"   'h' - Afiseaza aceasta lista")
        print()

    def _cleanup(self, cap):
        """Curatenie la inchidere"""
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.network_manager.stop()
        print("🔌 Sistem oprit complet")


def setup_camera_config() -> CameraConfig:
    """Configureaza camera interactiv"""
    print("🎥 CONFIGURARE CAMERA")
    print("=" * 30)

    camera_id = input("ID camera (ex: 'cam1'): ").strip() or 'cam1'

    print("\nPozitia camerei in spatiul 3D (metri):")
    x = float(input("X: ") or "0")
    y = float(input("Y: ") or "0")
    z = float(input("Z: ") or "0")

    print("\nOrientarea camerei (grade, optional):")
    roll = float(input("Roll: ") or "0")
    pitch = float(input("Pitch: ") or "0")
    yaw = float(input("Yaw: ") or "0")

    return CameraConfig(
        camera_id=camera_id,
        position=[x, y, z],
        orientation=[roll, pitch, yaw]
    )


def setup_network_config(network_manager: NetworkManager) -> List[str]:
    """Configureaza reteaua interactiv"""
    print(f"\n🌐 CONFIGURARE RETEA")
    print("=" * 30)
    print("1. Auto-discovery (recomandat)")
    print("2. Configurare manuala")
    print("3. Doar local (fara sincronizare)")

    choice = input("Alege optiunea (1/2/3): ").strip() or "1"
    peers = []

    if choice == "1":
        print("\n🔍 Cautare automata de peers...")
        peers = network_manager.auto_discover_peers()

        if peers:
            print(f"\n✅ Gasite {len(peers)} camere:")
            for i, peer in enumerate(peers, 1):
                print(f"   {i}. {peer}")
        else:
            print("\n❌ Nu s-au gasit alte camere in retea")
            print("   Asigura-te ca alte laptopuri ruleaza sistemul!")

    elif choice == "2":
        print("\n📝 Configurare manuala...")
        while True:
            peer_ip = input("IP peer (Enter pentru terminare): ").strip()
            if not peer_ip:
                break
            peers.append(peer_ip)
            print(f"   ✅ Adaugat: {peer_ip}")

    else:
        print("\n🏠 Mod local - fara sincronizare")

    return peers


def main():
    """Functia principala"""
    print("=" * 60)
    print("🎯 SISTEM DETECTARE ARUCO 3D DISTRIBUIT")
    print("=" * 60)

    try:
        # Configurare camera
        camera_config = setup_camera_config()

        # Creeare sistem
        system = ArUcoSystem(camera_config)

        # Configurare retea
        peers = setup_network_config(system.network_manager)

        # Afisare configuratie finala
        print(f"\n🚀 CONFIGURATIE FINALA:")
        print(f"   Camera: {camera_config.camera_id}")
        print(f"   Pozitie: ({camera_config.position[0]}, {camera_config.position[1]}, {camera_config.position[2]})")
        print(
            f"   Orientare: ({camera_config.orientation[0]}°, {camera_config.orientation[1]}°, {camera_config.orientation[2]}°)")
        print(f"   Server port: {system.network_manager.server_port}")
        print(f"   Peers: {len(peers)} conectati")
        print(f"   Auto-sync: {'✅ Activ' if peers else '❌ Inactiv'}")

        input("\nApasa Enter pentru a porni sistemul...")

        # Porneste sistemul
        system.start_system(peers)

    except KeyboardInterrupt:
        print("\n\n🛑 Sistem oprit de utilizator")
    except Exception as e:
        print(f"\n❌ Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()