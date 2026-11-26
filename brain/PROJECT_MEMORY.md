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
- [x] Konfirmasi Dosen: **Usulan Utama Disetujui Penuh** (Opsi Cadangan 1 disiapkan untuk fase lanjutan).
- [x] Setup isolasi Python Virtual Environment di Drive E: (`venv`).
- [x] Instalasi dependensi CV lengkap (`torch`, `torchvision`, `opencv-python`, `mediapipe`, `ultralytics`).
- [x] Deteksi Hardware: AMD Radeon™ Graphics (iGPU) — optimasi inferensi CPU via Velocity Filter divalidasi.
- [x] Implementasi kode modular `src/`:
  - `src/utils/box_utils.py`: Adaptive Dual-Hand Unified Bounding Box & 1:1 Square Unified ROI (`compute_square_unified_roi`, `crop_square_roi`).
  - `src/utils/geometry.py`: 3D joint angles, 174-dim Dual-Hand Feature Extractor (`extract_dual_hand_features`), & authentic BISINDO geometric disambiguation (`disambiguate_bisindo_pair` M vs N, B vs F).
  - `src/utils/camera_gui.py`: GUI Tkinter pemilih kamera & sumber input (*live preview*, *Mirror toggle*, *Browse Video/Citra/Folder*).
  - `src/modules/velocity_filter.py`: Velocity-Aware Transition Filter (Anti-ghost recognition).
  - `src/modules/dual_hand_detector.py`: MediaPipe Tasks HandLandmarker 42 keypoints extractor.
  - `src/modules/landmark_classifier.py`: PyTorch 42-Keypoint Dual-Hand Landmark MLP (`LandmarkMLP` + `LandmarkClassifierWrapper`).
  - `src/dataset/prep_cls_dataset.py`, `src/dataset/live_sim.py`, & `src/dataset/collector.py`: Stratified classification dataset prep + simulator kondisi webcam live (`flip`, `far`, `blur`, pencahayaan) + webcam calibration collector (`run_collect.bat`).
  - `src/pipeline.py`: Real-time multi-source Hybrid Ensemble pipeline (`0.60 YOLOv11n-cls + 0.40 Landmark MLP` + Temporal Smoothing + PiP Model ROI HUD).
  - `src/train_cls.py` & `src/train_landmarks.py`: Training pipeline (`run_train.bat`).
- [x] Penemuan dataset lokal di `E:\Semester 5\C&V\Dataset\data 1\Citra BISINDO` (26 kelas alfabet A-Z, 312 citra).
- [x] Upgrade Arsitektur ke Hybrid Ensemble v2.1 (Live-Robustness Training: 1.160 citra YOLO + 13.678 sampel Landmark MLP):
  - `YOLOv11n-cls` (20 epochs, imgsz=224, `fliplr=0.5`, 260 asli + 900 crop kondisi live): **Val Top-1 = 100,00% (52/52)**, latensi **5,5 ms**, bobot `weights/bisindo_yolov11n_cls_best.pt` (3,2 MB).
  - `Landmark MLP` (70 epochs, 13.678 sampel dengan mirror handedness swap + deteksi live-sim): **Val Top-1 = 96,08% (49/51)**, bobot `weights/bisindo_landmark_mlp.pt`.
  - Benchmark Ketahanan Live (`robust_eval`): `flip` **44% → 100%**, `far+flip` **38% → 88%**, `far` **81% → 88%**, `bright` **98% → 100%**, `motion blur` **100%** (LM **90% → 100%**).
- [x] Dokumentasi Lengkap Laporan Eksperimen & Plot Kurva: `brain/04_Implementation/Training_Report_YOLOv11n.md`.
- [x] Validasi Pengujian Real-Time Webcam (Single-Hand & Dual-Hand 42 Keypoints berhasil diuji pada Webcam 3): `brain/04_Implementation/Realtime_Testing_Report.md`.

---

## 6. Roadmap Sesi Berikutnya
1. **Fase Pengujian Real-Time End-to-End dengan Model Aktif**:
   - Menjalankan `run_pipeline.bat` untuk menguji inferensi langsung alfabet BISINDO dari webcam dengan bobot `weights/bisindo_yolov11n_best.pt`.
   - Menguji perangkaian huruf menjadi kata/kalimat di buffer teks.
   - Menguji keandalan modul disambiguasi M vs N secara langsung.
2. **Fase Penyusunan Naskah Laporan Akhir**:
   - Mengompilasi laporan eksperimen dan bukti visual ke dalam format laporan praktikum / artikel ilmiah lengkap.
3. **Fase Ekspansi Lanjutan (Opsional/Sesuai Arahan Dosen)**:
   - Integrasi Opsi Cadangan 1 (Buffer sequence Bi-LSTM untuk kata bergerak dinamis seperti huruf Z atau kata BISINDO).
