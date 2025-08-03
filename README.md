# 🎯 Sistem de Detectare ArUco 3D Distribuit

Sistem distribuit pentru detectarea markerilor ArUco și calculul poziției 3D folosind mai multe camere sincronizate.

## 🏗️ Arhitectura Sistemului

```
📁 aruco-3d-detection-system/
├── 📄 main.py                  # Aplicația principală
├── 📄 data_structures.py       # Structuri de date
├── 📄 detector.py              # Detectarea ArUco
├── 📄 triangulation.py         # Calcule 3D și triangulație
├── 📄 network_manager.py       # Comunicarea în rețea
├── 📄 synchronization.py       # Managementul sincronizării
├── 📄 visualization.py         # Vizualizare 3D și UI
├── 📄 requirements.txt         # Dependințe
└── 📄 README.md               # Această documentație
```

## 🎮 Funcționalități

### 🎯 Detectare ArUco
- Detectare în timp real a markerilor ArUco
- Suport pentru dicționare multiple (DICT_5X5_250 implicit)
- Calcul automat al distanței și unghiurilor
- Calibrare automată a matricei camerei

### 📐 Calcule 3D
- Triangulație 3D din multiple camere
- Filtrarea outlier-ilor pentru precizie
- Transformări de coordonate globale
- Calcul vectori de direcție în spațiul 3D

### 🌐 Comunicare Distribuită
- Auto-discovery de camere în rețeaua locală
- Sincronizare automată prin timestamp-uri
- Arhitectură peer-to-peer robustă
- Toleranță la erori de rețea

### 📊 Vizualizare
- Text mărit și negru cu contur alb pe markeri
- Informații live: ID, distanță, azimut, elevație, poziție 3D
- Vizualizare 3D interactivă cu matplotlib
- Overlay cu statistici de sistem

## 📦 Instalare

### 1. Clonează repository-ul
```bash
git clone https://github.com/USERNAME/aruco-3d-detection-system.git
cd aruco-3d-detection-system
```

### 2. Instalează dependințele
```bash
pip install -r requirements.txt
```

### 3. Verifică instalarea
```python
python -c "import cv2; print('OpenCV:', cv2.__version__); print('ArUco available:', hasattr(cv2, 'aruco'))"
```

## 🚀 Utilizare

### Rulare Simplă
```bash
python main.py
```

### Configurare Multi-Cameră

**Pe primul laptop (Camera Master):**
```bash
python main.py
```
- ID cameră: `cam1`
- Poziție: `(0, 0, 0)` 
- Alege "Auto-discovery"

**Pe al doilea laptop:**
```bash
python main.py
```
- ID cameră: `cam2`
- Poziție: `(1, 0, 0)` (1 metru pe axa X)
- Alege "Auto-discovery" → va găsi primul laptop automat

**Pe al treilea laptop:**
```bash
python main.py
```
- ID cameră: `cam3`
- Poziție: `(0, 1, 0)` (1 metru pe axa Y)
- Se va conecta automat la celelalte două

## 🎮 Comenzi în Timpul Rulării

| Tastă | Funcție |
|-------|---------|
| `q` | Ieșire din program |
| `v` | Vizualizare 3D interactivă |
| `c` | Afișare statistici complete |
| `f` | Toggle fullscreen |
| `r` | Informații rezoluție cameră |
| `s` | Salvează frame curent |
| `t` | Ajustează timp sincronizare |
| `h` | Afișează lista de comenzi |

## 🔧 Configurare Avansată

### Calibrarea Camerei
Pentru precizie maximă, calibrează camera cu imagini de tablă de șah:

```python
from detector import ArUcoDetector

detector = ArUcoDetector("cam1")
success = detector.calibrate_camera_from_chessboard("path/to/chessboard/images/")
```

### Ajustarea Parametrilor

**Schimbarea dicționarului ArUco:**
```python
# În detector.py, linia 25:
self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
```

**Ajustarea ferestrei de sincronizare:**
```python
# În synchronization.py sau prin tasta 't':
sync_manager.set_time_window(0.05)  # 50ms
```

**Modificarea dimensiunii markerului:**
```python
# În detector.py, linia 30:
self.marker_length = 0.08  # 8 cm
```

## 📊 Markeri ArUco

### Generarea Markerilor
1. **Online:** [chev.me/arucogen](https://chev.me/arucogen/)
2. **Python:**
```python
import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 23, 200)
cv2.imwrite('marker_23.png', marker_image)
```

### Recomandări Tipărire
- **Dimensiune minimă:** 3x3 cm
- **Recomandată:** 5x5 cm pentru distanțe normale
- **Bordură albă:** Minim 1 cm în jurul markerului
- **Suprafață plană:** Carton rigid, nu hârtie moale

## 📐 Formule Matematice

### Calculul Distanței
```
distanță = ||tvec|| = √(tx² + ty² + tz²)
```

### Calculul Unghiurilor
```
azimut = arctan2(ty, tx)
elevație = arctan2(tz, √(tx² + ty²))
```

### Triangulația 3D
```
Pentru două camere cu pozițiile P1, P2 și direcțiile D1, D2:
Punctul 3D = intersecția liniilor (P1 + t1*D1) și (P2 + t2*D2)
```

## 🔍 Debugging

### Probleme Comune

**Markerii nu sunt detectați:**
- Verifică lumina (evită umbrele puternice)
- Asigură-te că markerul e din dicționarul corect
- Verifică calitatea tipăririi

**Auto-discovery nu funcționează:**
- Dezactivează firewall-ul temporar
- Verifică că laptopurile sunt în aceeași rețea WiFi
- Folosește configurarea manuală cu IP-urile directe

**Triangulația e inexactă:**
- Calibrează camerele cu table de șah
- Măsoară precis pozițiile camerelor
- Asigură-te că markerii sunt pe suprafață plană

### Loguri de Debug
Sistemul afișează automat:
- 📡 Conexiuni de rețea
- 📨 Mesaje primite/trimise
- 🎯 Numărul de detectări
- ⚠️ Erori de comunicare

## 🤝 Contribuții

Pentru îmbunătățiri sau bug-uri:
1. Fork repository-ul
2. Creează un branch: `git checkout -b feature-improvement`
3. Commit: `git commit -m "Adaugă funcționalitatea X"`
4. Push: `git push origin feature-improvement`
5. Creează un Pull Request

## 📄 Licență

MIT License - vezi fișierul LICENSE pentru detalii.

## 🔗 Resurse Utile

- [Documentația OpenCV ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Generator markeri ArUco](https://chev.me/arucogen/)
- [Calibrarea camerelor](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Triangulația în Computer Vision](https://en.wikipedia.org/wiki/Triangulation_(computer_vision))

---

**Dezvoltat pentru detectarea și triangulația 3D distribuită a markerilor ArUco** 🎯