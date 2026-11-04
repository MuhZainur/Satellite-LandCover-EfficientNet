# Skenario Video: Klasifikasi Tutupan Lahan dari Citra Satelit

Durasi target 2:30 sampai 2:50. Bahasa Indonesia sehari-hari. Narasi untuk ElevenLabs, visual dengan Remotion oleh model lain.

## 1. Aturan yang wajib dipatuhi

1. Semua angka harus sama dengan file ini.
2. Hasil model yang boleh disebut sebagai kemampuan: akurasi 91,9 persen pada 160 gambar uji. Kepercayaan 97,5 persen tidak boleh disebut sebagai kemampuan.
3. Kisi 6x6 selalu disebut "36 petak acak dari dataset", jangan disebut "peta wilayah" atau "kawasan".
4. Jangan menulis "deteksi deforestasi" atau "HIGH URBANIZATION" di layar.
5. Gunakan hanya gambar yang ada di folder proyek.

## 2. Spesifikasi teknis

Remotion
- 1920x1080, 30 fps, keluaran mp4 (h264).
- Satu `<Sequence>` per scene. Durasi dari mp3 (`getAudioDurationInSeconds`) ditambah 0,5 detik. Timestamp di bawah perkiraan.
- Fade antar scene 10 frame. Teks layar fade dan geser naik 20 px selama 12 frame.
- Latar #0B1410, teks putih, aksen hijau (#4ADE80). Warna kelas di peta ikut gambar `regional_landcover_map.png`, jangan diganti. Judul 64 px, isi 36 px, margin 120 px.
- Teks layar adalah teks Remotion. Subtitle memakai teks narasi, maksimal dua baris.
- Salin aset ke `public/` dan panggil dengan `staticFile()`.

ElevenLabs
- Model `eleven_multilingual_v2`, suara Indonesia yang tenang. Stability 0,5, similarity 0,75, style 0.
- Satu file per scene: `s01.mp3` sampai `s07.mp3`.
- Ejaan khusus di narasi (misalnya "Gred Kam", "Sentinel Dua") jangan diubah.

## 3. Aset yang boleh dipakai

| Path di folder proyek | Dipakai untuk |
| :--- | :--- |
| `regional_landcover_map.png` | mosaik dan peta warna, scene 4 dan 6 |
| `confusion_matrix.png` | hasil uji, scene 5 |
| `tile_inspection_gradcam.png` | Grad-CAM, scene 4 |
| `landcover_distribution_chart.png` | grafik distribusi, scene 6 |
| `data/2750/` (satu gambar per kelas) | contoh kelas, scene 2 |
| `environmental_audit_report.json` | angka, scene 6 |

## 4. Scene

### Scene 1, 0:00 sampai 0:13, pembuka

Visual: judul "Mengenali Jenis Lahan dari Citra Satelit". Satu petak satelit zoom perlahan di latar.

Teks layar: Klasifikasi Tutupan Lahan, Sentinel-2

Narasi:
> Dari atas, lahan kelihatan seperti kotak-kotak warna. Di proyek ini, komputer belajar menebak jenis lahan dari potongan kecil citra satelit.

### Scene 2, 0:13 sampai 0:34, data dan kelas

Visual: grid sepuluh gambar kecil, satu per kelas, dengan label muncul bergantian: hutan, sungai, jalan raya, permukiman, dan seterusnya.

Teks layar: EuroSAT. 27.000 petak, 10 kelas, 64x64 piksel

Narasi:
> Datanya bernama Euro Sat, dari satelit Sentinel Dua. Isinya dua puluh tujuh ribu petak kecil berukuran enam puluh empat piksel, dengan sepuluh jenis lahan. Ada hutan, sungai, jalan raya, permukiman, kawasan industri, dan beberapa jenis lahan pertanian.

### Scene 3, 0:34 sampai 1:00, model dan pelatihan

Visual: diagram alur. Petak masuk ke kotak "EfficientNet B0, sudah pernah belajar dari ImageNet", keluar sepuluh batang peluang. Di bawahnya tiga kotak pembagian data: 1.120 latih, 220 validasi, 160 uji.

Teks layar: 1.500 gambar. 1.120 latih, 220 validasi, 160 uji

Narasi:
> Modelnya Efisien Net B nol, yang sebelumnya sudah belajar mengenali gambar secara umum, lalu saya sesuaikan untuk sepuluh jenis lahan. Dari tiap kelas saya ambil seratus lima puluh gambar. Hasilnya seribu seratus dua puluh untuk melatih, dua ratus dua puluh untuk validasi, dan seratus enam puluh untuk menguji.

### Scene 4, 1:00 sampai 1:28, Grad-CAM

Visual: `tile_inspection_gradcam.png`. Gambar asli di kiri, peta panas di kanan. Lingkaran menyorot area paling merah.

Teks layar: Grad-CAM: bagian gambar yang paling berpengaruh

Narasi:
> Supaya keputusan model tidak terasa seperti kotak hitam, saya tambahkan Gred Kam. Ia mengambil gradien dari lapisan terakhir jaringan, lalu menggambar peta panas di atas petak. Bagian yang paling merah adalah bagian yang paling mempengaruhi tebakan model.

### Scene 5, 1:28 sampai 1:58, hasil pengujian

Visual: `confusion_matrix.png` penuh layar. Zoom ke diagonal, lalu ke sel HerbaceousVegetation ke PermanentCrop (angka 2). Teks besar "91,9 persen".

Teks layar: Akurasi uji 91,9 persen (147 dari 160)

Narasi:
> Pada seratus enam puluh gambar uji, model benar seratus empat puluh tujuh kali, yaitu sembilan puluh satu koma sembilan persen. Kelas industri, tanaman permanen, dan danau atau laut benar semua. Kesalahan yang paling sering adalah rumput liar yang ditebak sebagai tanaman permanen, sebanyak dua gambar.

### Scene 6, 1:58 sampai 2:30, ringkasan 36 petak

Visual: `regional_landcover_map.png`, mosaik kiri dan peta warna kanan. Lalu `landcover_distribution_chart.png`. Angka muncul: 20 dari 36 petak vegetasi, indeks hijau 55,6 persen. Label kecil di sudut: "36 petak acak dari dataset".

Teks layar: 36 petak acak dari dataset. Indeks hijau 55,6 persen

Narasi:
> Untuk dashboard, saya mengambil tiga puluh enam petak acak dari dataset dan menyusunnya jadi kisi enam kali enam. Ini bukan peta wilayah sungguhan, hanya contoh cara menghitung ringkasan. Dari tiga puluh enam petak, dua puluh berkelas tumbuhan, jadi indeks tutupan hijaunya lima puluh lima koma enam persen.

### Scene 7, 2:30 sampai 2:50, batasan dan penutup

Visual: daftar dua poin, lalu judul penutup.

Teks layar: Belum mendeteksi perubahan antar waktu. Diuji pada data Eropa saja

Narasi:
> Model ini belum mendeteksi deforestasi, karena belum membandingkan citra dari dua waktu. Ia juga baru diuji pada data Eropa. Terima kasih sudah menonton.

## 5. Daftar periksa sebelum render

1. Angka cocok: 27.000, 10 kelas, 64x64, 1.120 latih, 220 validasi, 160 uji, 147 dari 160, 91,9 persen, 36 petak, 20 petak, 55,6 persen.
2. Tidak ada angka 97,5 persen, kata "deforestasi" sebagai kemampuan, atau "HIGH URBANIZATION".
3. Label "36 petak acak dari dataset" terlihat selama scene 6.
4. Subtitle sama dengan narasi. Total 2:30 sampai 2:50.
