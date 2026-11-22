# DOKUMEN PERENCANAAN MODIFIKASI PENELITIAN
**Mata Kuliah:** Computer Vision (C&V) — Semester 5  
**Topik:** Real-Time Sign Language Interpretation & Hand Gesture Keypoint Tracking  
**Status:** Draf Usulan Modifikasi Penelitian Mandiri  

---

## 1. Identitas Dokumen & Paper Acuan

### A. Rencana Judul Penelitian Usulan
> **"Implementasi Sistem Penerjemah Alfabet BISINDO Real-Time Berbasis Dual-Hand Keypoint MediaPipe dan YOLOv11 dengan Integrasi Velocity-Aware Filter dan Geometric Disambiguation"**  
> *(Real-Time Two-Handed BISINDO Alphabet Interpretation System Based on MediaPipe Dual-Keypoint and YOLOv11 with Velocity-Aware Filtering and Geometric Disambiguation)*

### B. Paper Acuan Utama (State-of-the-Art Baseline)
* **Judul Asli:** *Real-Time American Sign Language Interpretation Using Deep Learning and Keypoint Tracking*
* **Penulis:** Bader Alsharif, Easa Alalwany, Ali Ibrahim, Imad Mahgoub, Mohammad Ilyas
* **Publikasi:** *Sensors* (MDPI), Vol. 25, No. 7, Art. 2138, Terbit: 28 Maret 2025
* **Indeksasi:** Scopus (Q1/Q2), Web of Science (SCIE)
* **DOI:** [10.3390/s25072138](https://doi.org/10.3390/s25072138)

---

## 2. Latar Belakang & Analisis Keterbatasan (*Research Gap*)

### A. Kekuatan Paper Acuan
Paper Alsharif dkk. (2025) menyajikan arsitektur deteksi gestur tangan yang sangat efisien dengan keunggulan:
1. **Akurasi Sangat Tinggi:** Meraih **mAP@0.5: 98,2%**, Precision: 98,5%, Recall: 98,1%, dan F1-Score: 99,1%.
2. **Efisiensi Komputasi Ekstrem:** Memakai YOLOv11n (2,59 juta parameter, 6,5 GFLOPs) yang mampu berjalan lancar secara *real-time* di prosesor laptop standar (Intel Core i7) via webcam tanpa kartu grafis (GPU) diskrit.
3. **Ketahanan Visual:** Pemanfaatan titik sendi MediaPipe membuat sistem kebal terhadap gangguan latar belakang (*background clutter*) dan variasi warna kulit.

### B. Celah Masalah (*Research Gap*) yang Ditemukan
Meskipun berkinerja tinggi, sistem pada paper asli memiliki **tiga keterbatasan fatal**:
1. **Keterbatasan Spasial Satu Tangan (*Single-Hand Bias*):**
   * Paper asli hanya dirancang untuk *American Sign Language* (ASL) yang seluruh alfabetnya (A–Z) berbasis **satu tangan** (21 keypoints).
   * Bahasa Isyarat Indonesia (**BISINDO**) memiliki banyak huruf yang wajib menggunakan **dua tangan** (misal: B, D, M, N, P, Q). Algoritma *bounding box* asli akan gagal total (*break*) saat kedua tangan saling berinteraksi.
2. **Kelemahan Dinamika Video Stream (*Ghost Detection / False Positives on Transitions*):**
   * Pada video webcam kontinu (30 FPS), ketika tangan pengguna berpindah pose dari satu huruf ke huruf berikutnya, terdapat fase transisi gerak (durasi ~0,2–0,4 detik).
   * Pada fase transisi ini, bentuk tangan yang setengah membuka sering **salah terdeteksi (*false positive*) sebagai huruf acak lain** (misal saat berganti dari A ke B, sempat terbaca huruf C atau L).
3. **Ambiguitas Spasial pada Huruf yang Bentuknya Mirip (*Visual Ambiguity*):**
   * Terdapat beberapa pasangan huruf dalam alfabet isyarat yang memiliki bentuk dasar kepalan mirip dan hanya dibedakan oleh detail kecil selipan ibu jari (misal: **M vs N**, atau **A vs E**). Detektor berbasis *bounding box* citra murni kerap mengalami penurunan *confidence* atau berfluktuasi cepat (*flickering*).

---

## 3. Konsep Usulan Modifikasi Terpadu (*The Proposed Solution*)

Penelitian usulan ini mengintegrasikan **tiga pilar modifikasi terpadu** yang tetap mempertahankan efisiensi komputasi ringan:

```
[Frame Webcam] ──> [MediaPipe Dual-Hands: Ekstrak 42 Titik Sendi]
                          │
                          ▼
        ┌────────────────────────────────────────────────────────┐
        │ MODIFIKASI 1: Velocity-Aware Transition Filter (Waktu) │
        └────────────────────────────────────────────────────────┘
                          │
             Apakah tangan sedang bergerak aktif?
             ├── Ya (Transisi) ──> Tahan / Skip Frame (Cegah Huruf Hantu)
             └── Tidak (Stabil)
                          │
                          ▼
        ┌────────────────────────────────────────────────────────┐
        │ MODIFIKASI 2: Adaptive Unified Dual-Hand Bounding Box  │
        └────────────────────────────────────────────────────────┘
                          │
             Membungkus area 1 tangan atau 2 tangan secara dinamis
                          │
                          ▼
        ┌────────────────────────────────────────────────────────┐
        │ Klasifikasi Detektor YOLOv11 (Fine-Tuned BISINDO)      │
        └────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌────────────────────────────────────────────────────────┐
        │ MODIFIKASI 3: Geometric Disambiguation Module (Ruang)  │
        └────────────────────────────────────────────────────────┘
                          │
             Apakah hasil prediksi masuk kelompok huruf ambigu?
             ├── Tidak (Yakin) ──> Cetak Huruf ke Layar
             └── Ya (Ragu M/N) ──> Validasi Sudut Sendi MediaPipe ──> Output Pasti
```

---

## 4. Rincian Teknis Tiga Modifikasi

### Modifikasi 1: Velocity-Aware Transition Filter (Filter Domain Waktu)
* **Tujuan:** Menghilangkan *false positive* saat tangan bergerak berpindah pose antar-huruf.
* **Mekanisme Kerja:**
  1. Sistem melacak koordinat pergelangan tangan (*wrist landmark*, ID: 0) pada frame saat ini $(x_t, y_t)$ dan frame sebelumnya $(x_{t-1}, y_{t-1})$.
  2. Menghitung magnitudo kecepatan pergeseran tangan:
     $$v_t = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$$
  3. Mengaplikasikan ambang batas gerak (*motion threshold* $\tau_v$):
     $$\text{State} = \begin{cases} \text{TRANSISI (Skip YOLO)}, & \text{jika } v_t \ge \tau_v \\ \text{STABIL (Jalankan YOLO)}, & \text{jika } v_t < \tau_v \end{cases}$$
* **Keuntungan:** Mengeliminasi huruf sampah dan menghemat penggunaan daya CPU karena inferensi YOLO dihentikan sementara saat tangan bergerak.

---

### Modifikasi 2: Adaptive Unified Dual-Hand Bounding Box (Penanganan BISINDO)
* **Tujuan:** Mengakomodasi gestur huruf BISINDO yang melibatkan satu tangan maupun dua tangan.
* **Mekanisme Kerja:**
  1. MediaPipe diatur untuk mendeteksi hingga 2 tangan secara serentak (menghasilkan maksimal 42 titik sendi).
  2. Sistem memeriksa jarak Euclidean antar-kedua pergelangan tangan ($d_{\text{wrist}}$).
  3. Jika kedua tangan aktif dan berada dalam rentang interaksi gestur ($d_{\text{wrist}} < \text{threshold}$):
     Sistem menghitung **Unified Bounding Box** yang menyatukan titik ekstrem kedua tangan:
     $$\begin{aligned}
     X_{\min} &= \min(X_{\text{kiri}}, X_{\text{kanan}}), \quad Y_{\min} = \min(Y_{\text{kiri}}, Y_{\text{kanan}}) \\
     X_{\max} &= \max(X_{\text{kiri}}, X_{\text{kanan}}), \quad Y_{\max} = \max(Y_{\text{kiri}}, Y_{\text{kanan}})
     \end{aligned}$$
  4. Area kotak terpadu ini diumpankan ke model YOLOv11 yang di-*fine-tune* untuk 26 kelas BISINDO.

---

### Modifikasi 3: Geometric Disambiguation Module (Filter Domain Ruang)
* **Tujuan:** Menyelesaikan ambiguitas klasifikasi pada pasangan huruf yang secara visual tampak identik.
* **Mekanisme Kerja:**
  1. Didefinisikan himpunan pasangan huruf ambigu, contohnya $\mathcal{A} = \{(\text{M}, \text{N}), (\text{A}, \text{E})\}$.
  2. Jika prediksi YOLOv11 menghasilkan kelas dalam himpunan $\mathcal{A}$ dengan selisih probabilitas kecil ($|P_{\text{class1}} - P_{\text{class2}}| < 0.20$):
     Sistem memanggil modul verifikasi geometri koordinat 3D MediaPipe.
  3. Untuk kasus huruf **M vs N**:
     * Sistem menghitung sudut dan jarak relatif ujung ibu jari (landmark 4) terhadap pangkal jari manis (landmark 13) dan jari tengah (landmark 9):
       $$\theta_{\text{thumb}} = \arctan2(y_4 - y_0, x_4 - x_0)$$
     * Letak selipan ibu jari di bawah 3 jari (M) vs 2 jari (N) diputuskan secara deterministik matematis.

---

## 5. Matriks Perbandingan: Penelitian Asli vs Usulan

| Komponen Evaluasi | Penelitian Asli (Alsharif et al., 2025) | Penelitian Usulan Modifikasi |
|---|---|---|
| **Bahasa Isyarat** | American Sign Language (ASL) | Bahasa Isyarat Indonesia (BISINDO) |
| **Modalitas Tangan** | 100% Satu Tangan (Single-Hand) | Variasi Satu Tangan & Dua Tangan (Dual-Hand) |
| **Jumlah Titik Sendi** | 21 keypoints | Hingga 42 keypoints (2 × 21 titik) |
| **Penanganan Transisi** | Tidak ada (rentan *false alarm* saat tangan gerak) | **Velocity-Aware Transition Filter** (stabil) |
| **Pembeda Huruf Mirip** | Hanya mengandalkan *Softmax* YOLOv11 | **Hybrid Geometric Disambiguation** (YOLO + Sudut Sendi) |
| **Kebutuhan Komputasi** | Sangat Ringan (~14 FPS di CPU) | Tetap Sangat Ringan (operasi matematika O(1) tambahan) |

---

## 6. Rancangan Dataset & Metrik Pengujian

### A. Dataset yang Digunakan (Katalog Sumber Terverifikasi)
Terdapat 4 opsi sumber dataset BISINDO publik terkurasi yang siap digunakan sesuai kebutuhan:

1. **Roboflow Universe — BISINDO Sign Language Datasets (Format YOLO Ready)**
   * **Nama Repositori:** *BISINDO Sign Language* / *Bahasa Isyarat Alphabet BISINDO* di [universe.roboflow.com](https://universe.roboflow.com).
   * **Karakteristik:** Citra 26 alfabet BISINDO yang **sudah memiliki label bounding box (.txt)** siap pakai.
   * **Keunggulan Utama:** Dapat langsung diekspor 1-klik ke format `YOLOv8 / YOLOv11 PyTorch` lengkap dengan file `data.yaml`. Menghemat 100% waktu anotasi manual.

2. **Mendeley Data — Multimodal BISINDO Corpus & Landmark (Repositori Akademik Resmi)**
   * **Judul 1:** *A Multimodal BISINDO Corpus: Annotated Images and Videos for Indonesian Sign Language Hand Gesture Recognition* ([data.mendeley.com](https://data.mendeley.com)).
   * **Judul 2:** *Indonesian Sign Language (BISINDO) Dataset: Landmark Extraction for Capturing Non-Manual Markers* (Rhio Sutoyo dkk., Binus University).
   * **Karakteristik:** Dataset citra dan video alfabet A–Z pada berbagai variasi pencahayaan ruangan, dilengkapi file ekstraksi koordinat landmark `.npy` via MediaPipe.
   * **Keunggulan Utama:** Memiliki **DOI resmi** dan dapat disitasi secara akademik di laporan tugas/proposal skripsi.

3. **Kaggle — Bahasa Isyarat Indonesia (BISINDO) Alphabets**
   * **Nama Dataset:** *Bahasa Isyarat Indonesia (BISINDO) Alphabets* (oleh Achmad Noer dkk.).
   * **Karakteristik:** Ribuan citra gestur alfabet BISINDO terstruktur rapi ke dalam 26 folder kelas (A sampai Z).
   * **Keunggulan Utama:** Sangat populer, mudah diunduh secara bebas via Kaggle API / web.

4. **GitHub / Open Science — WL-BISINDO (Word-Level BISINDO Video Dataset)**
   * **Karakteristik:** Berisi 1.600 rekaman video RGB untuk 32 gestur kata/glos sehari-hari.
   * **Keunggulan Utama:** Ideal untuk menguji ketahanan modul *Velocity-Aware Transition Filter* pada aliran video kontinyu.

* **Pembagian Partisi Data:** **80% Data Latih (Train)**, **10% Data Validasi (Val)**, dan **10% Data Uji (Test)**.

### B. Metrik Pengujian yang Dihitung
1. **Akurasi Deteksi:** Precision ($P$), Recall ($R$), F1-Score, dan Mean Average Precision (**mAP@0.5**).
2. **Efektivitas Filter Transisi:** *False Positive Rate (FPR)* selama fase gerakan transisi tangan (ditargetkan turun >80% dibanding sistem tanpa filter).
3. **Akurasi Kelas Ambigu:** Akurasi khusus pada subset huruf mirip (M, N, A, E).
4. **Efisiensi Komputasi Real-Time:** *Latency per Frame* (ms) dan *Throughput* (**Frames Per Second / FPS**) pada CPU laptop standar.

---

## 7. Rencana Tahapan Pengerjaan (*Project Milestones*)

```
[Tahap 1: Studi Literatur & Perancangan Arsitektur] ──> Analisis matematis pipeline
                          │
                          ▼
[Tahap 2: Setup Lingkungan & Ekstraksi MediaPipe] ────> Implementasi Dual-Hand Tracking
                          │
                          ▼
[Tahap 3: Implementasi Modul Filter Ringan] ──────────> Koding Velocity Filter & Geometri Sudut
                          │
                          ▼
[Tahap 4: Fine-Tuning YOLOv11 & Evaluasi] ────────────> Uji akurasi, mAP, dan kecepatan FPS
                          │
                          ▼
[Tahap 5: Dokumentasi & Penyusunan Laporan Akhir] ────> Penulisan dokumen tugas matkul C&V
```

---

## 8. Ringkasan Nilai Kebaruan (*Novelty Justification*)

Apabila dosen menanyakan keabsahan modifikasi ini, dokumen perencanaan ini menegaskan bahwa:
> *"Penelitian usulan ini tidak sekadar mengganti data gambar dari ASL ke BISINDO, melainkan merancang arsitektur terintegrasi yang menyelesaikan **dua problem fundamental Computer Vision pada pemrosesan video interaktif**: (1) Mengatasi noise transisi gerak antar-frame menggunakan filter berbasis kecepatan waktu, dan (2) Mengatasi ambiguitas visual spasial dengan memadukan representasi deep learning konvolusional bersama verifikasi kinematika sendi geometris."*
