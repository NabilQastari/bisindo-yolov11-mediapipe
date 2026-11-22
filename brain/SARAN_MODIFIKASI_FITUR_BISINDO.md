# DOKUMEN SARAN MODIFIKASI FITUR TAMBAHAN (BISINDO)
**Mata Kuliah:** Computer Vision (C&V) — Semester 5  
**Paper Acuan:** Alsharif et al. (*Sensors MDPI, 2025*) — MediaPipe Hands + YOLOv11  
**Topik Penelitian:** Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO) Real-Time  

---

## 1. Fitur Tambahan Utama (Pilihan Utama)

### **Nama Fitur:**
> **Dual-Filter: Velocity-Aware Transition Filter & Geometric Disambiguation**

* **Masalah yang Diselesaikan:**
  1. *Ghost Detection (Huruf Hantu):* Saat tangan berpindah pose dari satu huruf ke huruf lain di depan webcam (30 FPS), bentuk tangan transisi sering keliru terdeteksi sebagai huruf acak lain (*false positive*).
  2. *Ambiguitas Huruf Mirip:* Detektor objek YOLOv11 sering ragu atau berfluktuasi pada pasangan huruf BISINDO yang bentuk kepalannya hampir identik (seperti **M vs N**, atau **A vs E**).

* **Cara Kerja Teknis:**
  * **Tahap 1 — Filter Waktu (Velocity Filter):** Sistem menghitung magnitudo pergeseran koordinat pergelangan tangan antar-frame:
    $$v_t = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$$
    Jika $v_t > \text{threshold}$ (tangan sedang bergerak aktif), proses inferensi YOLO dihentikan sementara (*skip frame*). Deteksi baru dieksekusi saat tangan sudah berhenti/stabil.
  * **Tahap 2 — Filter Ruang (Geometric Disambiguation):** Jika hasil prediksi YOLO menghasilkan probabilitas yang berdekatan pada kelas mirip (misal M vs N), sistem mengaktifkan validasi sudut sendi jempol dari MediaPipe (`math.atan2` atau jarak ke jari tengah) untuk menentukan kelas yang benar secara matematis.

* **Nilai Tambah Ilmiah (Computer Vision):**
  Menggabungkan pengolahan temporal (filter dinamika gerak video) dengan verifikasi geometri sendi 3D deterministik. Bebas huruf sampah dan hemat komputasi CPU.

---

## 2. Fitur Tambahan Cadangan 1 (Alternatif Lingkungan / Citra)

### **Nama Fitur:**
> **Adaptive Illumination & Shadow Equalizer (CLAHE in LAB Color Space)**

* **Masalah yang Diselesaikan:**
  Kamera webcam laptop sangat rentan mengalami penurunan akurasi drastis saat dioperasikan di ruangan dengan pencahayaan buruk: ruangan redup (*low light*), bayangan tajam pada jari, atau kondisi tangan membelakangi sumber cahaya (*backlight*).

* **Cara Kerja Teknis:**
  * Menambahkan modul pra-pengolahan citra (*preprocessing*) sebelum frame masuk ke MediaPipe dan YOLO.
  * Frame video RGB dikonversi ke ruang warna **LAB**.
  * Algoritma **CLAHE (*Contrast Limited Adaptive Histogram Equalization*)** diaplikasikan secara dinamis hanya pada kanal **L (Luminance / Terang-Gelap)**, sementara kanal warna A dan B tidak diubah:
    ```python
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    frame_ready = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    ```

* **Nilai Tambah Ilmiah (Computer Vision):**
  Meningkatkan ketahanan (*illumination robustness*) sistem penglihatan komputer terhadap variasi intensitas cahaya ruangan tanpa merusak representasi tekstur warna kulit asli.

---

## 3. Fitur Tambahan Cadangan 2 (Alternatif Skala / Jarak)

### **Nama Fitur:**
> **Dynamic Distance & Scale-Invariant Normalization**

* **Masalah yang Diselesaikan:**
  Akurasi model sering anjlok jika posisi duduk pengguna berubah-ubah terhadap kamera. Tangan yang terlalu dekat menyebabkan gambar terpotong (*clipped*), sedangkan tangan yang terlalu jauh menyebabkan ukuran telapak tangan menjadi sangat kecil dan kabur (*low-resolution target*).

* **Cara Kerja Teknis:**
  * Memanfaatkan koordinat kedalaman ($z$-axis) MediaPipe dan panjang referensi anatomis telapak tangan (jarak titik pergelangan ID:0 ke pangkal jari tengah ID:9):
    $$D_{\text{palm}} = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$
  * Nilai $D_{\text{palm}}$ digunakan sebagai faktor pembagi (skalar) untuk menormalisasi ukuran *bounding box* dan resolusi *crop* secara dinamis ke rasio kanonikal sebelum diumpankan ke model YOLOv11.

* **Nilai Tambah Ilmiah (Computer Vision):**
  Menghasilkan model deteksi yang kebal terhadap perubahan jarak kamera (*scale & distance-invariant*), sehingga pengguna bebas bergerak maju atau mundur dari laptop tanpa menurunkan akurasi.

---

## 4. Matriks Ringkasan Pemilihan Fitur

| Parameter | Fitur Utama | Cadangan 1 | Cadangan 2 |
|---|---|---|---|
| **Fokus Masalah** | Dinamika Video Stream & Ambiguitas Huruf | Kondisi Cahaya & Bayangan Ruangan | Variasi Jarak Pengguna ke Kamera |
| **Domain Solusi** | Temporal Gating + Geometri Sendi | Image Preprocessing (Histogram LAB) | Spatial Normalization (Skala 3D) |
| **Kompleksitas Koding** | Ringan (Perhitungan vektor $\Delta x, \Delta y$) | Sangat Ringan (3 baris OpenCV CLAHE) | Ringan (Perhitungan jarak Euclidean) |
| **Kelebihan Utama** | Menghilangkan huruf hantu saat tangan gerak | Sistem tetap jalan di ruangan remang/gelap | Akurasi stabil di jarak dekat maupun jauh |
