---
title: Review Paper Acuan - Alsharif et al. (2025)
created: 2026-11-22
tags:
  - literature-review
  - baseline-paper
  - mediapipe
  - yolov11
  - asl
aliases:
  - Alsharif2025
  - Paper Acuan Utama
---

# 📖 Review Paper Acuan: Alsharif et al. (2025)

## 📌 Metadata Publikasi
* **Judul:** *Real-Time American Sign Language Interpretation Using Deep Learning and Keypoint Tracking*
* **Penulis:** Bader Alsharif, Easa Alalwany, Ali Ibrahim, Imad Mahgoub, Mohammad Ilyas
* **Jurnal:** *Sensors* (MDPI), Vol. 25, No. 7, Art. 2138
* **Tahun:** Maret 2025
* **Indeksasi:** Scopus (Q1/Q2), Web of Science (SCIE)
* **DOI:** [10.3390/s25072138](https://doi.org/10.3390/s25072138)
* **Navigasi:** Kembali ke [[00_Dashboard]] | Lanjut ke [[Concept_Overview]]

---

## 🔍 Ringkasan Metodologi Asli
Paper ini merancang sistem *two-stage vision pipeline* untuk menerjemahkan alfabet American Sign Language (ASL A–Z):
1. **Google MediaPipe Hands:** Mengekstrak 21 titik koordinat 3D sendi tangan dari webcam laptop.
2. **Bounding Box Matematis:** Dihitung dari nilai minimum dan maksimum 21 koordinat $(x, y)$:
   $$X_{\min} = \min(x_0, \dots, x_{20}), \quad X_{\max} = \max(x_0, \dots, x_{20})$$
3. **YOLOv11n Transfer Learning:** Backbone dibekukan (*frozen*), neck dan head di-*fine-tune* pada dataset 130.000 citra seimbang (5.000 citra/kelas untuk 26 alfabet ASL).

---

## 📊 Hasil Performa
* **mAP@0.5:** 98,2%
* **Precision / Recall / F1:** 98,5% / 98,1% / 99,1%
* **Kompleksitas Komputasi:** 2,59 M parameter, 6,5 GFLOPs
* **Kecepatan:** 1,3 ms (GPU), ~72,5 ms / 13,8 FPS (CPU laptop Intel Core i7-1260P)

---

## ⚠️ Keterbatasan & Celah Riset (*Research Gap*)
1. **Single-Hand Constraint:** Hanya bisa membaca 1 tangan (ASL). Begitu diterapkan ke [[Concept_Overview|BISINDO]] yang menggunakan dua tangan, sistem gagal total.
2. **Ghost Detection (Transisi Gerak):** Tidak ada penanganan fase transisi gerak antar-pose, diselesaikan melalui [[Velocity_Aware_Filter]].
3. **Ambiguitas Huruf Mirip:** YOLOv11 sering ragu pada pasangan huruf yang bentuk kepalannya identik (M vs N), diselesaikan melalui [[Geometric_Disambiguation]].
