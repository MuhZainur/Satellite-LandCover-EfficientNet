# 🛰️ Regional Land Cover & Environmental Deforestation Monitoring (Geospatial AI)

Dashboard analitik tutupan lahan dan pemantauan deforestasi berbasis citra satelit **EU Copernicus Sentinel-2 (EuroSAT)** dan model **EfficientNet-B0**. Mengubah citra mosaik satelit resolusi tinggi menjadi peta tematik tutupan lahan 10-kelas (*thematic cartography*), menghitung indeks tutupan hijau (*Ecological Green Index*), serta menyediakan interpretasi keputusan model via **Grad-CAM** dan estimasi ketidakpastian **Shannon Entropy**.

---

## 🎯 Nilai Bisnis & Masalah Industri
- **Problem:** Mengapa klasifikasi satu tile satelit kecil (64x64) sering membingungkan dan tidak bernilai bisnis bagi pengambil kebijakan? Karena pemerintah daerah, dinas lingkungan hidup, atau perusahaan kehutanan tidak menginspeksi satu petak acak, melainkan membutuhkan **gambaran spasial lanskap regional**: berapa persen hutan yang masih lestari, ke mana arah ekspansi kawasan industri/pemukiman, dan apakah terjadi perambahan liar di sempadan sungai.
- **Solusi AI:**
  1. **Regional Spatial Survey:** Menganalisis mosaik $6 \times 6$ petak citra satelit (36 kawasan terhubung) untuk memetakan distribusi lahan regional secara menyeluruh.
  2. **Thematic Cartographic Mapping:** Menghasilkan peta tematik dengan pewarnaan standar tata ruang geospasial (Hutan: Hijau tua, Pemukiman: Merah bata, Badan Air: Biru, Industri: Abu-abu, Lahan Pertanian: Kuning/Krem).
  3. **Ecological Green Index (EGI):** Menghitung rasio tutupan vegetasi hijau terhadap total area yang disurvei sebagai indikator utama kepatuhan ESG lingkungan.
  4. **Interpretasi & Keamanan Model:** Visualisasi Grad-CAM dikombinasikan dengan metrik *Shannon Entropy* untuk mendeteksi tingkat keraguan model pada area batas geografis.

---

## 🏗️ Pendekatan Teknis (Intermediate-Level Engineering)
1. **Satellite Feature Backbone:** EfficientNet-B0 yang diadaptasi untuk 10 kelas tutupan lahan Sentinel-2:
   - `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
2. **Balanced Stratified Training:** Dilatih pada subset representatif 1.500 citra satelit multispektral yang seimbang antar kelas dengan regularisasi augmentasi spasial (random horizontal & vertical flips).
3. **Cartographic Mosaic Generation:** Mengumpulkan 36 petak citra satelit menjadi kisi $6 \times 6$, menjalankan inferensi batch, dan memplot peta tematik berdampingan dengan mosaik satelit asli lengkap dengan legenda warna.
4. **Explainability & Uncertainty Scoring:**
   - **Grad-CAM:** Memetakan gradien aktivasi lapisan konvolusi terakhir (`features[-1]`) ke ruang citra satelit.
   - **Shannon Entropy:** $H(P) = -\sum_{i=1}^{10} p_i \log_2(p_i)$ mengukur kejelasan keputusan model (skor rendah = model sangat yakin dan dapat diandalkan).

---

## 📊 Hasil Benchmark & Validasi Riil

Pengujian dijalankan pada dataset resmi **Copernicus Sentinel-2 (EuroSAT)**:
- **Total Citra Dataset:** 27.000 citra satelit multispektral (10 kelas)
- **Dataset Evaluasi Uji:** 1.500 citra seimbang (150 citra per kelas)
- **Akurasi Keseluruhan (Test Accuracy):** **91.9%** across all 10 classes

| Kelas Tutupan Lahan | Precision | Recall | F1-Score | Keterangan Evaluasi |
| :--- | :---: | :---: | :---: | :--- |
| **Forest (Hutan Alami)** | 0.887 | 0.920 | **0.903** | Delineasi vegetasi rapat sangat tinggi |
| **Residential (Pemukiman)** | 0.979 | 0.957 | **0.968** | Pola atap bangunan terdeteksi presisi |
| **SeaLake (Danau / Laut)** | 0.985 | 0.955 | **0.970** | Hampir sempurna membedakan badan air |
| **Pasture (Padang Rumput)** | 0.978 | 0.978 | **0.978** | Akurasi sangat konsisten |
| **Industrial (Zona Industri)** | 0.842 | 0.941 | **0.889** | Struktur fasilitas manufaktur terpetakan |
| **Highway (Jalan Raya)** | 0.880 | 0.880 | **0.880** | Jaringan konektivitas jalan |
| **River (Aliran Sungai)** | 0.824 | 0.857 | **0.840** | Meander sungai berliku |

---

## 🖼️ Hasil Visual & Laporan Lingkungan

- **Artikel Publikasi Teknis (Medium Style):** [`MEDIUM_ARTICLE.md`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/MEDIUM_ARTICLE.md) / [`artikel.md`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/artikel.md)  
  Ulasan engineering mendalam: penskalaan EfficientNet-B0, interpretabilitas geospasial Grad-CAM, kuantifikasi ketidakpastian entropi Shannon di zona ekoton, dan agregasi mosaik regional.
- **Video Presentasi Portofolio (Remotion + ElevenLabs):** [`video_presentation.mp4`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/video_presentation.mp4) ([Naskah Skenario Bahasa Inggris](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/skenario_video_en.md))  
  Video presentasi Full HD 1080p berdurasi ~2:22 menit dengan narasi suara natural berbahasa Inggris, teks layar, dan demonstrasi survei mosaik satelit 6x6.
- **Peta Mosaik Regional Tematik:** [`regional_landcover_map.png`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/regional_landcover_map.png)  
  Survei spasial $6 \times 6$ petak satelit Sentinel-2 berdampingan dengan peta kartografi klasifikasi AI 10-warna beserta legenda resmi.
- **Distribusi Tutupan Lahan & Green Index:** [`landcover_distribution_chart.png`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/landcover_distribution_chart.png)  
  Diagram batang sebaran luas peruntukan lahan serta skor *Ecological Green Index* (**55.6% Green Cover**).
- **Inspeksi Grad-CAM & Entropi:** [`tile_inspection_gradcam.png`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/tile_inspection_gradcam.png)  
  Analisis transparansi keputusan model pada citra satelit dengan skor ketidakpastian Shannon Entropy (**0.040 bit** = sangat stabil).
- **Confusion Matrix 10-Kelas:** [`confusion_matrix.png`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/confusion_matrix.png)
- **Laporan Audit Lingkungan Hidup:** [`environmental_audit_report.json`](file:///H:/coding/Portfolio/Computer%20Vision/06_Satellite_LandCover_Classifier/environmental_audit_report.json)  
  Dokumen JSON audit yang merinci luas per kelas, Green Index, dan ringkasan metrik F1-score model.

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Survei Regional & Menghasilkan Dashboard
```bash
python demo.py
```
Perintah ini akan memuat bobot `model.pt`, melakukan survei spasial $6 \times 6$ petak citra satelit Sentinel-2, menghitung *Ecological Green Index*, menganalisis Grad-CAM dan entropi, lalu menyimpan semua gambar visual (`regional_landcover_map.png`, `landcover_distribution_chart.png`, `tile_inspection_gradcam.png`, `environmental_audit_report.json`).

### 2. Melatih Ulang Model EfficientNet-B0
```bash
python train.py --epochs 4 --batch-size 32
```
