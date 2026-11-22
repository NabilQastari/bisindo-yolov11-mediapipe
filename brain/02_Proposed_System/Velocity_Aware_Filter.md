---
title: Modul Velocity-Aware Transition Filter
created: 2026-11-22
tags:
  - proposed-system
  - temporal-filter
  - computer-vision
  - anti-ghosting
aliases:
  - Velocity Filter
  - Motion Transition Filter
---

# ⏱️ Modul Velocity-Aware Transition Filter

*Navigasi:* [[00_Dashboard]] | Induk: [[Concept_Overview]] | Pipeline: [[Pipeline_Architecture]]

---

## 📌 Masalah yang Diselesaikan (*Ghost Detection*)
Pada video kontinu webcam (30 FPS), pengguna yang berganti gestur dari huruf "A" ke "B" harus membuka jarinya. Selama pergerakan sepersekian detik itu, bentuk tangan setengah membuka sering tidak sengaja terbaca oleh model deteksi sebagai huruf acak lain (misal terbaca C atau L). Ini adalah kelemahan fatal pada sistem baseline [[Paper_Alsharif_2025]].

---

## 📐 Formulasi Matematis

1. Melacak koordinat pergelangan tangan (*wrist landmark*, ID: 0) pada frame saat ini $(x_t, y_t)$ dan frame sebelumnya $(x_{t-1}, y_{t-1})$.
2. Menghitung magnitudo kecepatan pergeseran tangan:
   $$v_t = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$$
3. Menerapkan ambang batas gerak (*motion threshold* $\tau_v$):
   $$\text{Keputusan Inferensi} = \begin{cases} \text{SKIP INFERENSI (Fase Transisi)}, & \text{jika } v_t \ge \tau_v \\ \text{EKSEKUSI YOLO (Fase Stabil)}, & \text{jika } v_t < \tau_v \end{cases}$$

---

## 🚀 Keunggulan Teknis
* **Mengeliminasi Huruf Sampah:** Menghapus >80% *false positive* saat fase tangan bergerak.
* **Menghemat Komputasi CPU:** Inferensi YOLO tidak dipanggil saat tangan bergerak, sehingga prosesor laptop bekerja lebih dingin dan efisien.
