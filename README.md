# 🇮🇩 Real-Time BISINDO Sign Language Interpretation System
### Powered by MediaPipe Dual-Keypoint Tracking, YOLOv11, and Adaptive Filtering

[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-blue.svg)](https://github.com/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/Keypoint-Google%20MediaPipe-FF6F00.svg)](https://developers.google.com/mediapipe)
[![Obsidian Vault](https://img.shields.io/badge/Knowledge%20Base-Obsidian-7C3AED.svg)](https://obsidian.md)
[![Graphify](https://img.shields.io/badge/Graph-Graphify-green.svg)](https://github.com/)

---

## 📌 Deskripsi Proyek
Proyek ini mengimplementasikan sistem penerjemah alfabet **Bahasa Isyarat Indonesia (BISINDO)** secara *real-time* berbasis kamera web laptop standar. Sistem ini merupakan pengembangan dan modifikasi arsitektural dari paper acuan bereputasi internasional:
> **Alsharif et al. (2025)** — *"Real-Time American Sign Language Interpretation Using Deep Learning and Keypoint Tracking"*, Sensors (MDPI), DOI: [10.3390/s25072138](https://doi.org/10.3390/s25072138).

---

## 🚀 Fitur Utama & Modifikasi Ilmiah
1. **Adaptive Dual-Hand Tracking:** Mengonfigurasi MediaPipe untuk melacak hingga 42 titik sendi secara simultan dan merumuskan *Unified Bounding Box* untuk mengakomodasi gestur dua tangan khas alfabet BISINDO (seperti huruf B, D, M, N, P, dan Q).
2. **Velocity-Aware Transition Filter:** Filter domain waktu berbasis kecepatan gerak pergelangan tangan untuk mengeliminasi *false positive* / huruf hantu saat tangan berpindah pose.
3. **Geometric Disambiguation Module:** Verifikasi sudut dan jarak sendi jempol deterministik secara geometris untuk membedakan huruf dengan bentuk fisik mirip (M vs N).
4. **Lightweight & Real-Time:** Dirancang hemat daya (~15 FPS pada CPU murni laptop standar) tanpa ketergantungan pada GPU mahal.

---

## 📂 Struktur Repositori
* **`brain/`** — **Obsidian Vault:** Memuat seluruh dokumentasi modular, kajian literatur, dan desain arsitektur yang saling terhubung dengan *wikilinks* `[[...]]`.
* **`graphify/`** — **Knowledge Graph:** Berisi visualisasi graf interaktif `graph.html` hasil pemetaan Graphify.
* **`src/`** — Kode sumber implementasi pipeline Computer Vision di Python.
* **`run_graphify.bat`** — Skrip otomatis 1-klik untuk meng-generate graf proyek.

---

## ⚡ Panduan Membuka Dokumentasi di Obsidian
1. Unduh dan buka aplikasi [Obsidian](https://obsidian.md).
2. Pilih **"Open folder as vault"**.
3. Arahkan ke folder **`brain/`** di repositori ini.
4. Buka file `00_Dashboard.md` dan tekan `Ctrl + G` untuk melihat *Graph View* relasi antar-modul penelitian.

---

## 🛠️ Instalasi & Menjalankan Dependensi
```bash
# Clone repositori ini
git clone https://github.com/USERNAME/bisindo-yolov11-mediapipe.git
cd bisindo-yolov11-mediapipe

# Buat virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```
