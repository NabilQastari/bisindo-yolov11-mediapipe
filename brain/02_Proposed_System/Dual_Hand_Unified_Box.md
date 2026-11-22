---
title: Modul Adaptive Unified Dual-Hand Bounding Box
created: 2026-11-22
tags:
  - proposed-system
  - spatial-module
  - dual-hand
  - bisindo
aliases:
  - Dual Hand Box
  - Unified Bounding Box
---

# 🖐️ Modul Adaptive Unified Dual-Hand Bounding Box

*Navigasi:* [[00_Dashboard]] | Induk: [[Concept_Overview]] | Pipeline: [[Pipeline_Architecture]]

---

## 📌 Masalah yang Diselesaikan
Pada paper acuan [[Paper_Alsharif_2025]], sistem mengasumsikan input hanya satu tangan (ASL). Pada **BISINDO**, banyak huruf membutuhkan **dua tangan** (seperti B, D, M, N, P, dan Q). Detektor satu tangan akan gagal ketika kedua tangan saling bersentuhan atau bertumpuk di depan dada.

---

## ⚙️ Mekanisme Logika Algoritma

1. **Deteksi Multi-Hand:** Google MediaPipe dikonfigurasi dengan parameter `max_num_hands=2`, menghasilkan maksimal $2 \times 21 = 42$ titik sendi 3D.
2. **Pengecekan Interaksi Spasial:** Sistem mengukur jarak Euclidean antarsendi pergelangan tangan:
   $$d_{\text{wrist}} = \sqrt{(x_{\text{wrist\_L}} - x_{\text{wrist\_R}})^2 + (y_{\text{wrist\_L}} - y_{\text{wrist\_R}})^2}$$
3. **Percabangan Bounding Box:**
   * **Jika 1 Tangan Aktif (Huruf A, C, E, L):** Menghitung *Single Bounding Box* dari 21 titik tangan yang aktif.
   * **Jika 2 Tangan Aktif & Berdekatan (Huruf B, D, M, N, P):** Menghitung **Unified Bounding Box** yang menyatukan titik ekstrem kedua tangan:
     $$\begin{aligned}
     X_{\min} &= \min(X_{\text{L}}, X_{\text{R}}), \quad Y_{\min} = \min(Y_{\text{L}}, Y_{\text{R}}) \\
     X_{\max} &= \max(X_{\text{L}}, X_{\text{R}}), \quad Y_{\max} = \max(Y_{\text{L}}, Y_{\text{R}})
     \end{aligned}$$
4. **Normalisasi & Cropping:** Area kotak terpadu di-*crop* dan dinormalisasi untuk diumpankan ke model YOLOv11 yang di-*fine-tune* pada dataset [[BISINDO_Datasets_Catalog]].
