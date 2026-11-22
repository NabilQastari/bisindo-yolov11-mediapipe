---
title: Arsitektur Pipeline Sistem Keseluruhan
created: 2026-11-22
tags:
  - implementation
  - pipeline
  - computer-vision
  - integration
aliases:
  - Pipeline Sistem
  - Arsitektur Sistem
---

# 🏗️ Arsitektur Pipeline Sistem Keseluruhan

*Navigasi:* [[00_Dashboard]] | Konsep: [[Concept_Overview]] | Baseline: [[Paper_Alsharif_2025]]

---

## 🔄 Alur Pemrosesan Frame Real-Time

```mermaid
flowchart TD
    A["Kamera Webcam (640x640)"] --> B["MediaPipe Dual-Hands<br>(Ekstrak 42 Keypoints)"]
    B --> C["[[Velocity_Aware_Filter]]<br>(Hitung Kecepatan Pergelangan)"]
    
    C -- "Kecepatan > Ambang (Tangan Gerak)" --> D["Skip Frame / Hold Last Letter<br>(Cegah Huruf Hantu)"]
    C -- "Kecepatan < Ambang (Tangan Stabil)" --> E["[[Dual_Hand_Unified_Box]]<br>(Hitung Kotak 1 atau 2 Tangan)"]
    
    E --> F["Crop & Normalisasi Area Tangan"]
    F --> G["Klasifikasi YOLOv11n<br>(Dataset [[BISINDO_Datasets_Catalog]])"]
    
    G --> H{"Apakah Prediksi Ambigu?<br>(M vs N / A vs E)"}
    H -- "Tidak (Confidence Tinggi)" --> J["Tampilkan Huruf"]
    H -- "Ya (Ragu-ragu)" --> I["[[Geometric_Disambiguation]]<br>(Cek Sudut Sendi Jempol)"]
    I --> J
    
    J --> K["Buffer Pengetikan & UI Interaktif"]
```

---

## ⚡ Estimasi Latensi Komputasi per Frame (CPU Laptop i7)
1. **Preprocessing:** ~1,1 ms
2. **MediaPipe Tracking (42 Titik):** ~12,0 ms
3. **Velocity Filter Calculation:** ~0,1 ms
4. **Bounding Box Calculation:** ~0,1 ms
5. **YOLOv11n Inference (jika fase stabil):** ~55,0 ms *(Jika fase gerak, YOLO di-skip $\rightarrow$ latensi 0 ms!)*
6. **Geometric Disambiguation (opsional):** ~0,2 ms
7. **Total Latensi Rata-rata:** ~68,5 ms (**~15 FPS pada CPU murni laptop standar**)
