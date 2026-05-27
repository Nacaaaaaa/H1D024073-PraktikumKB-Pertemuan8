# Dokumentasi Praktikum Convolutional Neural Network - Gunting Batu Kertas

**Identitas Mahasiswa:**
* Nama: Nalendra Wicaksana
* NIM: H1D024073
* Shift Lama: E
* Shift Baru; F

## 1. Tujuan Penugasan
Membangun, melatih, serta menganalisis keandalan model arsitektur Convolutional Neural Network (CNN) untuk mendeteksi, mengekstraksi matriks spasial visual, dan mengklasifikasikan aset citra digital ke dalam kategori bentuk tangan gunting, batu, dan kertas secara otomatis.

## 2. Struktur Repositori
- rockpaperscissors/ : Direktori dataset gambar yang terbagi atas folder paper, rock, dan scissors.
- main_rps.py : Berisi kode implementasi pengolahan citra digital dan arsitektur CNN.
- README.md : Dokumentasi teknis mengenai mekanisme kerja program komputer.

## 3. Penjelasan Kerja Kode dan Dampak Output

### Kode Inisialisasi Prapemrosesan Citra (ImageDataGenerator):
- train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#### Penjelasan Kode:
[cite_start]Baris ini menginisialisasi objek 'ImageDataGenerator' yang bertindak sebagai pemroses awal otomatis data citra [cite: 318-325]. Parameter 'rescale' mengalikan setiap nilai piksel gambar asli dengan matriks pecahan 1/255, sedangkan parameter 'validation_split' mencadangkan porsi data gambar sebesar 20 persen dari total keseluruhan untuk dialokasikan menjadi data uji validasi.
#### Dampak Output:
Rentang nilai warna piksel gambar yang semula bernilai 0 sampai 255 dikompresi secara seragam menjadi skala desimal dinamis antara 0.0 hingga 1.0 agar proses kalkulasi gradien bobot jaringan berjalan lebih stabil.

Kode Integrasi Aliran Data Direktori:
- train_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(150, 150), batch_size=32, class_mode='categorical', subset='training'
)
#### Penjelasan Kode:
[cite_start]Mengaktifkan fungsi pembacaan data dinamis langsung dari struktur folder fisik komputer [cite: 326-339]. Parameter 'target_size' memaksa seluruh gambar masukan diubah dimensinya menjadi resolusi spasial 150x150 piksel, 'batch_size' membatasi pasokan data per iterasi, dan 'class_mode' diatur ke bentuk 'categorical' karena klasifikasi menargetkan tiga kelompok kategori kelas.
#### Dampak Output:
Terminal akan menampilkan log konfirmasi tekstual otomatis yang menyatakan jumlah total berkas gambar yang berhasil terdeteksi beserta jumlah kelas kategori folder yang ditemukan pada direktori penyimpanan.

### Kode Lapisan Ekstraksi Fitur (Konvolusi dan Pooling):
- Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
MaxPooling2D(2,2),
#### Penjelasan Kode:
[cite_start]Lapisan 'Conv2D' menerapkan operasi konvolusi spasial dua dimensi menggunakan 32 filter matriks persegi berukuran 3x3 piksel untuk mendeteksi pola tepi, sudut, dan tekstur objek gambar tangan [cite: 346-347]. Fungsi aktivasi 'ReLU' mengeliminasi nilai perhitungan bernilai negatif. Lapisan 'MaxPooling2D' melakukan penyusutan dimensi gambar dengan mengambil nilai piksel maksimum pada matriks sampling ukuran 2x2.
#### Dampak Output:
Ukuran matriks representasi gambar mengecil secara signifikan namun informasi esensial penanda visual bentuk tangan tetap dipertahankan, mengurangi beban kalkulasi parameter komputer.

### Kode Lapisan Perata dan Klasifikasi Akhir:
- Flatten(),
Dense(512, activation='relu'),
Dense(3, activation='softmax')
#### Penjelasan Kode:
[cite_start]Fungsi 'Flatten' meratakan susunan dimensi matriks peta fitur multidimensi hasil proses konvolusi menjadi struktur vektor satu dimensi linier [cite: 354-355]. [cite_start]Vektor tersebut diteruskan menuju lapisan 'Dense' tersembunyi berkapasitas 512 neuron, dan diakhiri lapisan penentu klasifikasi dengan 3 unit neuron berbasis fungsi probabilitas 'Softmax' [cite: 270-271].
#### Dampak Output:
Data visual bertransformasi penuh menjadi angka keputusan probabilitas biner. Model siap mengeluarkan perkiraan kelas dominan yang menjadi representasi tebakan gambar gunting, batu, atau kertas.
