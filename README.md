# indonesian-food-image-classification
End-to-end PyTorch pipeline untuk klasifikasi 15 makanan Indonesia menggunakan transfer learning (ConvNeXt/ResNet/EfficientNet), data cleaning+augmentasi, pseudo-labeling semi-supervised (CNN+PCA+KMeans), early stopping &amp; best model, serta otomatisasi file matching &amp; submission.csv.


Program ini adalah pipeline deep learning end-to-end untuk klasifikasi citra makanan Indonesia (15 kelas) menggunakan transfer learning ConvNeXt-Tiny/ResNet50/EfficientNet-B0 di PyTorch, lengkap dengan tahap data cleaning sampai pembuatan file submission akhir.

1.	Ayam Bakar
2.	Ayam Betutu
3.	Ayam Goreng
4.	Ayam Pop
5.	Bakso
6.	Coto Makassar
7.	Gado Gado
8.	Gudeg
9.	Nasi Goreng
10.	Pempek
11.	Rawon
12.	Rendang
13.	Sate Madura
14.	Sate Padang
15.	Soto

Preprocessing data, membersihkan dan memvalidasi label dari CSV, memperbaiki nama kolom, serta memastikan seluruh path gambar train valid sebelum training.

Data augmentation (flip, rotation, color jitter, affine) dan konfigurasi hyperparameter terpusat (learning rate, weight decay, optimizer, image size, dll.).

Metode semi-supervised learning dengan pseudo-labeling berbasis fitur CNN, PCA, dan K-Means untuk memanfaatkan data tanpa label secara terkontrol menggunakan confidence threshold.

Training loop dengan early stopping, scheduler, penyimpanan best model, serta visualisasi kurva loss/accuracy dan confusion matrix.

“Intelligent file matching” untuk test set (mencocokkan ID di test.csv dengan file gambar di folder), prediksi batch, analisis distribusi prediksi, dan generator submission.csv yang mengikuti urutan asli test.csv sehingga siap di-submit ke platform kompetisi.
