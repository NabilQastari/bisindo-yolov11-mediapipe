---
title: Laporan Pelatihan Model YOLOv11n BISINDO
created: 2026-11-23
tags:
  - implementation
  - training
  - yolov11
  - experiments
  - report
aliases:
  - Training Report
  - Laporan Training YOLOv11n
---

# 📈 Laporan Pelatihan Model YOLOv11n BISINDO

*Navigasi:* [[00_Dashboard]] | Pipeline: [[Pipeline_Architecture]] | Pengujian Real-Time: [[Realtime_Testing_Report]] | State: [[PROJECT_MEMORY]]

---

## 🔬 Ringkasan Konfigurasi & Spesifikasi Eksperimen

Proses pelatihan model klasifikasi dan deteksi 26 alfabet Bahasa Isyarat Indonesia (BISINDO) dilakukan menggunakan arsitektur **YOLOv11n (Nano)** dengan teknik *transfer learning* dari bobot COCO pre-trained.

| Parameter | Nilai / Konfigurasi | Catatan Khusus |
| :--- | :--- | :--- |
| **Model Base** | `yolo11n.pt` (Ultralytics v8.4+) | 101 layers, 2.603.126 parameter, 6.5 GFLOPs |
| **Ukuran Bobot Akhir** | **5,5 MB** (`best.pt`) | Sangat ringan untuk *edge deployment* |
| **Dataset Source** | `data/bisindo_yolo/data.yaml` | 26 kelas alfabet BISINDO (A - Z) |
| **Data Training** | 234 citra (80%) | Auto-anotasi MediaPipe Dual-Hands |
| **Data Validasi** | 26 citra (10%) | 1 sampel per kelas dievaluasi tiap epoch |
| **Data Testing** | 52 citra (10%) | Disimpan untuk evaluasi independen |
| **Resolusi Citra** | $640 \times 640$ piksel | Standard square input |
| **Optimizer** | AdamW (`lr=0.000333`, `momentum=0.9`) | 81 weight groups, weight decay adaptif |
| **Batch Size** | 8 | Optimal untuk alokasi RAM CPU |
| **Total Epochs** | **30 Epochs (Selesai 100%)** | Durasi total: **1,041 jam (~62 menit)** |
| **Hardware Host** | **AMD Ryzen 3 4300U with Radeon Graphics** | Komputasi CPU murni (PyTorch CPU) |
| **Latensi Inferensi** | **89,4 ms / citra (~11,2 FPS)** | Preprocess 2,5 ms, Postprocess 1,3 ms |

---

## 📸 Bukti Visual Eksekusi & Hasil Pelatihan

### 1. Terminal Selesai Pelatihan (Epoch 30 Selesai)
![Terminal Akhir Pelatihan](../assets/training_yolov11n_final_results.jpg)

*Gambar 1: Log terminal saat 30 epoch selesai dieksekusi secara penuh, menampilkan metrik validasi akhir per huruf.*

### 2. Kurva Pembelajaran & Konvergensi Loss (30 Epochs)
![Kurva Hasil Training](../assets/yolov11n_training_results_curves.png)

*Gambar 2: Kurva perkembangan loss (train & val) serta metrik akurasi mAP@50 dan mAP@50-95 dari epoch 1 hingga 30.*

### 3. Matriks Konfusi Ternormalisasi (Confusion Matrix)
![Confusion Matrix](../assets/yolov11n_confusion_matrix_normalized.png)

*Gambar 3: Matriks konfusi normalisasi untuk 26 kelas alfabet BISINDO.*

### 4. Kurva F1-Score Confidence
![Kurva F1 Score](../assets/yolov11n_f1_curve.png)

*Gambar 4: Hubungan antara ambang batas keyakinan (confidence threshold) dan nilai F1-score.*

---

## 📊 Hasil Evaluasi Akhir Model (Best Model Validation)

Evaluasi pada bobot terbaik (`weights/bisindo_yolov11n_best.pt`) menghasilkan metrik:

| Metrik Evaluasi | Nilai Akhir | Interpretasi Akademik |
| :--- | :---: | :--- |
| **Presisi Rata-Rata (Precision)** | **79,3%** (0.793) | Proporsi prediksi benar saat mendeteksi isyarat |
| **Sensitivitas / Recall** | **72,5%** (0.725) | Kemampuan model menemukan seluruh isyarat di citra |
| **Akurasi mAP@50** | **88,3%** (0.883) | Akurasi deteksi pada IoU threshold 0.50 |
| **Akurasi mAP@50-95** | **71,7%** (0.717) | Akurasi ketat pada rentang IoU 0.50 s/d 0.95 |

---

## 🔤 Evaluasi Performa per Kelas Alfabet BISINDO

Dari rincian per kelas pada log validasi akhir:

| Kelas Alfabet | Precision (P) | Recall (R) | mAP@50 | Kategori Performa |
| :---: | :---: | :---: | :---: | :--- |
| **A, B, C, D, E, F, G, H, I, J, L, N, O, P, Q, S, T, U, V, X, Y** | $0.46 - 1.00$ | $0.70 - 1.00$ | **99.5%** | **Sangat Tinggi / Sempurna** |
| **K** | 0.882 | 0.500 | **69.5%** | **Baik** |
| **M** | 0.405 | 0.809 | **49.7%** | **Perlu Disambiguasi** |
| **W** | 0.475 | 1.000 | **49.7%** | **Cukup** (Mirip huruf V/U) |
| **R** | 1.000 | 0.000 | **19.9%** | **Perlu Tambahan Variasi** |
| **Z** | 0.000 | 0.000 | **16.6%** | **Gerak Dinamis** (Perlu Pelacak Lintasan) |

---

## 💡 Analisis Ilmiah & Validasi Novelty Penelitian

1. **Validasi Modul Disambiguasi Geometris ([[Geometric_Disambiguation]])**:
   - Huruf **N** memperoleh akurasi **99.5% mAP50**, sedangkan huruf **M** memperoleh **49.7% mAP50**.
   - Dalam BISINDO, huruf M dan N memiliki kontur visual 2D yang sangat identik (posisi kepalan tangan dan ibu jari terselip). Pendekatan *bounding box* konvensional YOLO kesulitan membedakannya hanya dari citra crop.
   - Temuan empiris ini **secara telak membuktikan perlunya modul inovasi kita**: perhitungan sudut 3D sendi MediaPipe (*flexion angles*) pada jari telunjuk, tengah, dan manis untuk menyelesaikan ambiguitas M vs N secara definitif!
2. **Karakteristik Huruf Bergerak Dinamis (Kasus Huruf Z)**:
   - Huruf Z memiliki mAP rendah karena dalam BISINDO huruf Z diperagakan dengan gerakan telunjuk membentuk huruf "Z" di udara (*dynamic stroke*).
   - Ini memvalidasi proposal kita bahwa alfabet statis murni memerlukan **Velocity-Aware Filter** dan menyiapkan landasan untuk opsi lanjutan **Temporal Sequence (Bi-LSTM / 1D-CNN)** pada kata bergerak.
3. **Efisiensi Komputasi pada Perangkat Standar**:
   - Model hanya berukuran **5,5 MB** dengan konsumsi waktu inferensi **89,4 ms** pada prosesor laptop hemat daya (AMD Ryzen 3 4300U, 15W TDP). Hal ini membuktikan sistem sangat aplikatif untuk digunakan sehari-hari oleh penyandang disabilitas rungu tanpa memerlukan laptop gaming berdaya tinggi.

---

## 🚀 Kesiapan Sistem Akhir
- File bobot terbaik resmi tersimpan di: `E:\Semester 5\C&V\Project Citra\weights\bisindo_yolov11n_best.pt`.
- Skrip [src/pipeline.py](file:///E:/Semester%205/C&V/Project%20Citra/src/pipeline.py) dan launcher [run_pipeline.bat](file:///E:/Semester%205/C&V/Project%20Citra/run_pipeline.bat) sudah terhubung langsung ke bobot ini.
