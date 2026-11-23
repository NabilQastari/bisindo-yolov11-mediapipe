---
title: Laporan Pengujian Real-Time Webcam Pipeline BISINDO
created: 2026-11-23
tags:
  - testing
  - real-time
  - webcam
  - mediapipe
  - velocity-filter
  - report
aliases:
  - Realtime Testing Report
  - Laporan Uji Real-Time
---

# 📹 Laporan Pengujian Real-Time Webcam Pipeline BISINDO

*Navigasi:* [[00_Dashboard]] | Baseline: [[Paper_Alsharif_2025]] | Pipeline: [[Pipeline_Architecture]] | Laporan Training: [[Training_Report_YOLOv11n]]

---

## 🎯 Tujuan Pengujian
Memvalidasi integrasi fungsional dari:
1. **MediaPipe Tasks HandLandmarker** dalam mengekstraksi 21 dan 42 titik sendi secara real-time.
2. **Velocity-Aware Transition Filter** dalam mengukur kecepatan gerak pergelangan tangan ($\vec{v}$) dan mengidentifikasi fase kestabilan isyarat (`STABLE` vs `TRANSITION`).
3. **Adaptive Bounding Box** dalam mengisolasi ROI (*Region of Interest*) tangan tunggal maupun kedua tangan.
4. **Multi-Source Input Selector** dalam menghubungkan kamera eksternal (Webcam 3) pada resolusi $1280 \times 720$.

---

## 🖼️ Bukti Eksperimen 1: Pengujian Isyarat Satu Tangan (Single-Hand)

![Uji Coba Real-Time Satu Tangan](../assets/realtime_testing_single_hand.jpg)

*Gambar 1: Tangkapan layar pengujian real-time pada isyarat tangan terbuka (Webcam 3). Sistem berhasil mengekstraksi 21 landmark anatomi dengan bounding box adaptif.*

### Analisis Hasil Pengujian 1:
- **Mode Bounding Box**: `single_right` (menandakan algoritma mengenali keberadaan satu tangan dominan kanan).
- **Status Filter Kecepatan**: `STABLE (Speed: 0.03)` — nilai kecepatan jauh di bawah ambang batas transisi ($0.40$), menandakan tangan berada pada posisi diam/mantap untuk siap diklasifikasikan.
- **Visual Landmark**: 21 titik sendi (titik oranye) dan sendi pergelangan (*wrist*, titik merah) terpetakan dengan akurasi spasial tinggi pada kontur telapak dan jari.
- **Performa Real-Time**: Berjalan pada **~5.2 FPS** pada komputasi murni CPU laptop standar.

---

## 🖼️ Bukti Eksperimen 2: Pengujian Isyarat Dua Tangan Simultan (Dual-Hand 42 Keypoints)

![Uji Coba Real-Time Dua Tangan](../assets/realtime_testing_dual_hand.jpg)

*Gambar 2: Pengujian simultan dua tangan (42 keypoints) menggunakan MediaPipe Tasks dan Adaptive Bounding Box.*

### Analisis Hasil Pengujian 2:
- **Mode Bounding Box**: `separated_dual` — karena jarak antar-pergelangan tangan ($d$) melebihi ambang batas interaksi ($0.45$), sistem secara cerdas membuat dua kotak ROI terpisah untuk masing-masing tangan.
- **Pelacakan Simultan 42 Keypoint**:
  - Tangan kanan pengguna: 21 keypoint dengan kode warna oranye.
  - Tangan kiri pengguna: 21 keypoint dengan kode warna hijau.
  - Kedua pergelangan tangan ditandai dengan lingkaran merah sebagai referensi pelacak kecepatan.
- **Status Filter Kecepatan**: `STABLE (Speed: 0.01)` — filter mengukur kecepatan kedua pergelangan secara paralel dan memvalidasi bahwa kedua tangan berada dalam kondisi stabil.
- **Performa Real-Time**: Berjalan mulus pada **~6.8 FPS** tanpa adanya penurunan *frame-rate* yang signifikan meskipun beban pelacakan meningkat dua kali lipat (42 titik sendi).

---

## 📌 Kesimpulan & Relevansi untuk Laporan Akhir

Hasil pengujian ini secara empiris membuktikan bahwa:
1. **Novelty Usulan Terbukti Berfungsi**: Integrasi MediaPipe Dual-Hands dan Filter Kecepatan berhasil diimplementasikan di lingkungan fisik nyata (*live camera feed*).
2. **Kestabilan Algoritma**: Tidak terjadi *flickering* atau *landmark loss* meskipun terdapat variasi latar belakang ruangan (*ambient background*).
3. **Kesiapan Integrasi YOLOv11**: Begitu pelatihan bobot `bisindo_yolov11n_best.pt` selesai, model deteksi akan langsung membaca area kotak hijau yang sudah tervalidasi ini untuk menghasilkan output teks alfabet secara langsung.
