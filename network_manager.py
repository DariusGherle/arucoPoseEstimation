"""
Modulul pentru managementul comunicarii in retea intre camere
"""

import socket
import json
import threading
import time
from typing import List, Dict, Callable, Optional
from dataclasses import asdict

from data_structures import ArUcoDetection, NetworkMessage


class NetworkManager:
    """Clasa pentru managementul conexiunilor de retea"""

    def __init__(self, camera_id: str, server_port: int = 8888):
        self.camera_id = camera_id
        self.server_port = server_port
        self.server_socket: Optional[socket.socket] = None
        self.running = False

        # Callback pentru procesarea detectiilor primite
        self.detection_callback: Optional[Callable[[str, List[ArUcoDetection]], None]] = None

        # Lista de peers conectati
        self.connected_peers: List[str] = []

    def set_detection_callback(self, callback: Callable[[str, List[ArUcoDetection]], None]):
        """Seteaza callback-ul pentru procesarea detectiilor primite"""
        self.detection_callback = callback

    def start_server(self):
        """Porneste serverul pentru a primi conexiuni"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind(('0.0.0.0', self.server_port))
            self.server_socket.listen(5)
            self.running = True

            print(f"🌐 Server pornit pe portul {self.server_port}")

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"📡 Conexiune noua de la {addr[0]}")

                    # Creaza thread pentru client
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.error:
                    if self.running:
                        print("⚠️ Eroare la acceptarea conexiunii")
                    break

        except Exception as e:
            print(f"❌ Eroare la pornirea serverului: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _handle_client(self, client_socket: socket.socket, addr):
        """Gestioneaza un client conectat"""
        peer_ip = addr[0]

        try:
            # Adauga peer-ul la lista
            if peer_ip not in self.connected_peers:
                self.connected_peers.append(peer_ip)

            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break

                try:
                    # Decodifica mesajul JSON
                    message_data = json.loads(data.decode('utf-8'))

                    # Proceseaza mesajul
                    self._process_message(message_data, peer_ip)

                except json.JSONDecodeError:
                    print(f"⚠️ Mesaj JSON invalid de la {peer_ip}")
                except Exception as e:
                    print(f"⚠️ Eroare la procesarea mesajului de la {peer_ip}: {e}")

        except socket.error:
            pass
        finally:
            # Curata conexiunea
            if peer_ip in self.connected_peers:
                self.connected_peers.remove(peer_ip)
            client_socket.close()
            print(f"📡 Conexiune inchisa cu {peer_ip}")

    def _process_message(self, message_data: dict, sender_ip: str):
        """Proceseaza un mesaj primit"""
        try:
            if 'detections' in message_data:
                camera_id = message_data['camera_id']
                detections_data = message_data['detections']

                # Converteste inapoi la obiecte ArUcoDetection
                detections = []
                for det_data in detections_data:
                    detection = ArUcoDetection(**det_data)
                    detections.append(detection)

                # Apeleaza callback-ul daca exista
                if self.detection_callback:
                    self.detection_callback(camera_id, detections)

                print(f"📨 Primit {len(detections)} detectii de la {camera_id} ({sender_ip})")

        except Exception as e:
            print(f"⚠️ Eroare la procesarea mesajului: {e}")

    def send_detections_to_peer(self, peer_ip: str, peer_port: int,
                                detections: List[ArUcoDetection]) -> bool:
        """Trimite detectiile la un peer"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(1.0)  # Timeout de 1 secunda
            client_socket.connect((peer_ip, peer_port))

            # Pregateste mesajul
            message = {
                'camera_id': self.camera_id,
                'detections': [asdict(det) for det in detections],
                'timestamp': time.time()
            }

            # Trimite datele
            json_data = json.dumps(message).encode('utf-8')
            client_socket.send(json_data)
            client_socket.close()

            return True

        except socket.error as e:
            print(f"⚠️ Eroare la trimiterea catre {peer_ip}:{peer_port} - {e}")
            return False

    def auto_discover_peers(self, broadcast_port: int = 8889) -> List[str]:
        """Descoperă automat alte camere în rețeaua locală"""

        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                return local_ip
            except:
                return "192.168.1.100"

        local_ip = get_local_ip()
        network_prefix = '.'.join(local_ip.split('.')[:-1]) + '.'

        print(f"🔍 Scanez rețeaua {network_prefix}0/24...")

        peers = []
        # Scanează IP-urile din rețeaua locală
        for i in range(1, 255):
            ip = f"{network_prefix}{i}"
            if ip == local_ip:
                continue

            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(0.05)  # 50ms timeout pentru scan rapid
                result = test_socket.connect_ex((ip, self.server_port))
                test_socket.close()

                if result == 0:
                    peers.append(ip)
                    print(f"  ✅ Găsit peer: {ip}")

            except:
                pass

        return peers

    def start_auto_sync(self, peers: List[str]):
        """Pornește sincronizarea automată cu peers"""

        def sync_thread(detection_source):
            while self.running:
                detections = detection_source()
                if detections:
                    # Trimite la toți peers
                    for peer_ip in peers:
                        self.send_detections_to_peer(peer_ip, self.server_port, detections)

                time.sleep(0.1)  # Sync la fiecare 100ms

        # Callback-ul va fi setat din exterior
        self._sync_thread_func = sync_thread

    def start_sync_with_detection_source(self, detection_source: Callable[[], List[ArUcoDetection]],
                                         peers: List[str]):
        """Porneste sincronizarea cu o sursa de detectii"""

        def sync_thread():
            while self.running:
                detections = detection_source()
                if detections:
                    for peer_ip in peers:
                        self.send_detections_to_peer(peer_ip, self.server_port, detections)

                time.sleep(0.1)

        sync_thread_obj = threading.Thread(target=sync_thread)
        sync_thread_obj.daemon = True
        sync_thread_obj.start()

    def stop(self):
        """Opreste toate conexiunile"""
        self.running = False

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("🔌 Network manager oprit")

    def get_connected_peers(self) -> List[str]:
        """Returneaza lista de peers conectati"""
        return self.connected_peers.copy()

    def is_peer_connected(self, peer_ip: str) -> bool:
        """Verifica daca un peer este conectat"""
        return peer_ip in self.connected_peers