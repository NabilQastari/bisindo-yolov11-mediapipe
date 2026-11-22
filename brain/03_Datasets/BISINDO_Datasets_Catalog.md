---
title: Katalog Dataset BISINDO Terverifikasi
created: 2026-11-22
tags:
  - datasets
  - bisindo
  - open-data
  - benchmark
aliases:
  - Dataset BISINDO
  - Katalog Data
---

# 📊 Katalog Dataset BISINDO Terverifikasi

*Navigasi:* [[00_Dashboard]] | Konsep: [[Concept_Overview]] | Pipeline: [[Pipeline_Architecture]]

---

## 🗂️ Sumber Dataset Terkurasi & Siap Pakai

### 1. Roboflow Universe — BISINDO Sign Language (Rekomendasi untuk YOLO)
* **Karakteristik:** Ribuan citra 26 alfabet BISINDO yang **sudah memiliki label bounding box format `.txt`**.
* **Keunggulan:** Bisa diekspor 1-klik langsung ke format `YOLOv8 / YOLOv11 PyTorch` lengkap dengan file `data.yaml`. Menghemat 100% waktu pelabelan manual.
* **Akses:** [universe.roboflow.com](https://universe.roboflow.com) (Ketik: *"BISINDO"*).

### 2. Mendeley Data — Multimodal BISINDO Corpus (Rekomendasi untuk Sitasi Laporan)
* **Karakteristik:** Memiliki **DOI resmi**, mencakup variasi pencahayaan ruangan dan file koordinat landmark `.npy` via Google MediaPipe (Rhio Sutoyo dkk., Binus University).
* **Akses:** [data.mendeley.com](https://data.mendeley.com/) (Ketik: *"BISINDO"*).

### 3. Kaggle — BISINDO Alphabets (Achmad Noer)
* **Karakteristik:** Citra terstruktur rapi ke dalam 26 folder kelas (A sampai Z).
* **Akses:** [kaggle.com](https://www.kaggle.com/) (Ketik: *"BISINDO Alphabets"*).

### 4. GitHub — WL-BISINDO (Word-Level BISINDO Video Dataset)
* **Karakteristik:** 1.600 rekaman video RGB untuk menguji ketahanan modul [[Velocity_Aware_Filter]] pada aliran video kontinu.

---

## 📐 Standar Partisi Data Latih & Uji
* **Data Latih (Training):** 80%
* **Data Validasi (Validation):** 10%
* **Data Uji (Testing):** 10%
