# 🕸️ Panduan Setup & Eksekusi Graphify
**Project:** Sistem Penerjemah BISINDO Real-Time  
**Lokasi Project:** `E:\Semester 5\C&V\Project Citra`  

---

## 📌 Apa Fungsi Folder Ini?
Folder `graphify` ini didedikasikan untuk menyimpan konfigurasi, skrip otomatisasi, serta hasil kompilasi knowledge graph dari **Graphify** (`graphifyy` di PyPI) yang memetakan seluruh catatan di folder `brain/` (Obsidian Vault) dan skrip kodingan Python ke dalam bentuk graf pengetahuan visual.

---

## 🚀 Cara Menjalankan Graphify

### 1. Instalasi (Cukup 1 Kali)
Buka terminal **PowerShell** atau **CMD**, lalu jalankan:
```bash
pip install graphifyy
```

### 2. Generate Knowledge Graph dari Vault `brain/`
Jalankan perintah berikut dari folder `Project Citra`:
```bash
cd "E:\Semester 5\C&V\Project Citra"
graphify ./brain --output ./graphify
```
Atau cukup klik dua kali file **`run_graphify.bat`** yang sudah disediakan di folder ini.

---

## 📂 Hasil Output Graphify
Setelah dieksekusi, Graphify akan menghasilkan:
* **`graph.html`** $\rightarrow$ Graf interaktif 2D/3D yang bisa dibuka di browser Chrome / Edge.
* **`GRAPH_REPORT.md`** $\rightarrow$ Peta struktur arsitektural proyek.
* **`graph.json`** $\rightarrow$ Basis data graf lokal.
