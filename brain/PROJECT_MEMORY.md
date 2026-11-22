# PROJECT MEMORY: BISINDO Hand Gesture Recognition (YOLOv11 + MediaPipe)

> **Dokumen Memory Proyek & State Tracker**  
> Disusun untuk memelihara konteks lintas sesi AI Assistant dan pengembangan semi-otonom.  
> Terakhir diperbarui: 23 November 2026.

---

## 1. Ringkasan Eksekutif & Identitas Proyek
* **Judul Proyek**: Adaptasi & Modifikasi Framework MediaPipe + YOLOv11 untuk Pengenalan Bahasa Isyarat Indonesia (BISINDO) Berbasis Isyarat Dua Tangan dan Filter Transisi Kecepatan.
* **Paper Acuan Utama**:  
  * *Alsharif et al.* (Sensors, MDPI, Jan 2025) – *"Robust Real-Time Hand Gesture Recognition Framework for American Sign Language Using MediaPipe and YOLOv11"*.
* **Repositori GitHub**: [https://github.com/NabilQastari/bisindo-yolov11-mediapipe.git](https://github.com/NabilQastari/bisindo-yolov11-mediapipe.git)
* **Branch Utama**: `main` (Remote Tracking terkonfigurasi aktif).
* **Workspace Path**: `E:\Semester 5\C&V\Project Citra`
* **Prinsip Operasional**: Semi-Autonomous (Pengelolaan Git commits, pushes, pembaruan knowledge graph 99% di-handle agent).
* **Aturan Instalasi**: **Eksklusif di Partisi E:** (Dilarang menginstal atau menaruh cache paket Python di Partisi C:).

---

## 2. Inovasi & Modifikasi Ilmiah

### Gap Penelitian dari Paper Acuan
Paper acuan (Alsharif et al., 2025) dirancang untuk **ASL (American Sign Language)** yang bersifat satu tangan (*single-hand* statis) dengan ROI individual per tangan. Pada konteks **BISINDO (Bahasa Isyarat Indonesia)**:
1. Mayoritas huruf (misal: B, D, G, P, T, X) dan kata melibatkan **interaksi kedua tangan secara bersamaan**.
2. ROI terpisah per tangan sering memecah konteks spasial saat kedua tangan bersentuhan (*overlap*).
3. Pengujian *live camera video stream* menghasilkan false-positive (*ghost detection*) saat tangan bertransisi dari satu isyarat ke isyarat lain.
4. Huruf dengan bentuk tangan mirip (seperti M vs N) sulit dibedakan hanya dari bounding box 2D.

### Usulan Modifikasi Utama (Active Proposal)
1. **Adaptive Dual-Hand Unified Bounding Box (42 Keypoints)**:
   * Menggunakan MediaPipe Hands dalam mode dual-hand.
   * Menghitung jarak Euclidean antar-wrist. Jika jarak di bawah threshold interaksi, sistem secara adaptif membentuk *unified bounding box* yang membungkus kedua tangan sebagai satu konteks spasial untuk YOLOv11.
2. **Velocity-Aware Transition Filter (Anti-Ghost Recognition)**:
   * Menghitung vektor kecepatan pergerakan landmark ($\vec{v} = \frac{\Delta \text{coords}}{\Delta t}$).
   * Ketika kecepatan tangan tinggi (fase perpindahan posisi), inferensi klasifikasi di-*suppress* dan baru diaktifkan kembali saat kestabilan spasial tercapai ($v < v_{\text{threshold}}$ selama $N$ frame).
3. **Geometric Disambiguation Module (MediaPipe 3D Joint Angles)**:
   * Menghitung sudut fleksi sendi PIP, DIP, dan MCP secara 3D untuk membedakan huruf beroklusi tinggi atau ambigu secara visual (misal: jumlah jari yang terlipat pada huruf M vs N).

### Rencana Cadangan (Backup Modifications)
* **Cadangan 1 (Temporal Dynamic Motion Recognition)**:
  * Menggunakan sliding window buffer (15–30 frame) yang dihubungkan ke lightweight sequence head (Bi-LSTM / GRU / 1D-CNN) untuk mengenali kata/frasa bergerak BISINDO (bukan hanya alfabet statis).
* **Cadangan 2 (Lightweight Edge Device Deployment & Quantization)**:
  * Optimasi model YOLOv11n ke INT8 quantization menggunakan TensorRT/ONNX Runtime untuk deployment pada edge devices (Raspberry Pi 5 / Jetson Nano / Android mobile) dengan target latency $\le 20$ ms.

---

## 3. Lingkungan & Konfigurasi Teknis

| Komponen | Spesifikasi / Path | Catatan Penting |
| :--- | :--- | :--- |
| **OS** | Windows | Powershell shell |
| **Python** | Python 3.12 (Base runtime) | Terletak di AppData host |
| **Virtual Env** | `E:\Semester 5\C&V\Project Citra\venv` | Seluruh package berada di drive E: |
| **Pip Cache** | `--no-cache-dir` | Menghindari pengotoran C:\Users AppData |
| **Graphify CLI** | `E:\Semester 5\C&V\Project Citra\venv\Scripts\graphify.exe` | Versi 0.9.115 |
| **Batch Helper** | `E:\Semester 5\C&V\Project Citra\run_graphify.bat` | Otomasi eksekusi graphify berbasis venv E: |

---

## 4. Struktur Direktori Proyek

```text
E:\Semester 5\C&V\Project Citra\
├── .gitignore                          # Filter file besar, dataset raw, venv, cache
├── README.md                           # Dokumentasi umum repositori GitHub
├── requirements.txt                    # Daftar dependensi pustaka
├── run_graphify.bat                    # Skrip 1-klik sinkronisasi Graphify
├── venv/                               # Virtual Environment Python di Partisi E:
├── brain/                              # Vault Obsidian (Dokumen & Knowledge Base)
│   ├── 00_Dashboard.md                 # Map of Content (MOC)
│   ├── PROJECT_MEMORY.md               # File memory lintas sesi (File ini)
│   ├── PLANNING_MODIFIKASI_PENELITIAN.md # Detail arsitektur & milestone
│   ├── SARAN_MODIFIKASI_FITUR_BISINDO.md # Ringkasan usulan fitur utama & cadangan
│   ├── USULAN_MODIFIKASI_PENELITIAN_BISINDO.docx # Berkas Word resmi untuk dosen
│   ├── 01_Literature_Review/
│   │   └── Paper_Alsharif_2025.md
│   ├── 02_Proposed_System/
│   │   ├── Concept_Overview.md
│   │   ├── Dual_Hand_Unified_Box.md
│   │   ├── Velocity_Aware_Filter.md
│   │   └── Geometric_Disambiguation.md
│   ├── 03_Datasets/
│   │   └── BISINDO_Datasets_Catalog.md
│   └── 04_Implementation/
│       └── Pipeline_Architecture.md
└── graphify/                           # Interactive Knowledge Graph Artifacts
    ├── graph.html                      # Web viewer visual interaktif
    ├── graph.json                      # Raw graph dataset (75 nodes, 90 edges)
    ├── GRAPH_REPORT.md                 # Analisis sentralitas & kluster modul
    └── README_GRAPHIFY.md              # Petunjuk integrasi
```

---

## 5. Status Progres Terakhir (Checklist)
- [x] Review & komparasi 12 jurnal Computer Vision & Visi Komputer.
- [x] Pemilihan paper benchmark (*Alsharif et al., Sensors MDPI 2025*).
- [x] Perumusan modifikasi ilmiah spesifik BISINDO (Dual-Hand Unified Box + Velocity Filter + Geometric Disambiguation).
- [x] Pembuatan dokumen resmi usulan penelitian dalam format `.docx` dan Markdown.
- [x] Perumusan 2 opsi fitur modifikasi cadangan (Temporal Motion Sequence & Edge INT8 Quantization).
- [x] Penyusunan Obsidian Vault `brain` lengkap dengan link relasi antar-catatan.
- [x] Setup isolasi Python Virtual Environment di Drive E:.
- [x] Instalasi paket `graphifyy` dan pembuatan knowledge graph interaktif.
- [x] Inisialisasi Git, branch `main`, binding remote origin GitHub `https://github.com/NabilQastari/bisindo-yolov11-mediapipe.git`.
- [x] Push perdana repositori ke GitHub (Clean tree & tracking aktif).

---

## 6. Roadmap Sesi Berikutnya
1. **Fase Konfirmasi Akademik**:
   * Evaluasi feedback dosen terhadap berkas `USULAN_MODIFIKASI_PENELITIAN_BISINDO.docx`.
   * Keputusan akhir: lanjut dengan Usulan Utama atau switch ke Usulan Cadangan 1 / 2.
2. **Fase Dataset & Preprocessing**:
   * Menyiapkan folder `data/` (di partisi E:).
   * Download & split dataset BISINDO (Kaggle BISINDO 26 Alfabet / Roboflow).
   * Scripting generator bounding box adaptif (MediaPipe 42 landmarks $\rightarrow$ YOLO format `.txt`).
3. **Fase Pipeline Development**:
   * Implementasi modul `dual_hand_detector.py`.
   * Implementasi modul `velocity_filter.py`.
   * Setup training script YOLOv11 nano/small dengan transfer learning.
