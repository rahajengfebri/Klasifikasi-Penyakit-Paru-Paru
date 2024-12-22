Nama : Rahajeng Febri Shafiyah

NIM : 202110370311308

Machine Learning A

# Klasifikasi Penyakit Paru-Paru
  Penyakit paru-paru merupakan salah satu masalah kesehatan yang umum terjadi, dipengaruhi oleh berbagai faktor seperti kebiasaan merokok, kurangnya aktivitas fisik, dan gaya hidup yang tidak sehat. Deteksi dini terhadap risiko penyakit paru-paru sangat penting untuk mencegah komplikasi serius serta meningkatkan kualitas hidup seseorang. Namun, proses diagnosis awal sering kali membutuhkan pemeriksaan medis yang memakan waktu dan biaya. Dengan kemajuan teknologi, khususnya di bidang kecerdasan buatan (Artificial Intelligence), pengembangan aplikasi prediksi risiko penyakit paru-paru menjadi solusi yang dapat membantu masyarakat dan tenaga medis. Aplikasi berbasis AI ini dirancang untuk memberikan gambaran awal mengenai potensi risiko penyakit paru-paru berdasarkan data pribadi, seperti usia, jenis kelamin, kebiasaan merokok, dan riwayat kesehatan.
  
## Dataset
  Proyek ini menggunakan dataset dari Kaggle (https://www.kaggle.com/datasets/andot03bsrc/dataset-predic-terkena-penyakit-paruparu) yang terdiri dari 30.000 data. Dengan memanfaatkan model prediksi seperti Feedforward Neural Network (FFNN) dan Random Forest, analisis data dapat dilakukan dengan lebih cepat dan akurat. Proyek ini bertujuan untuk memprediksi apakah seseorang berisiko menderita penyakit paru-paru berdasarkan data pribadi yang dimasukkan. Hasil prediksi ini diharapkan dapat mendukung upaya pencegahan serta membantu pengambilan langkah awal dalam menjaga kesehatan. Dataset ini memiliki dua kelas, yaitu "Tidak" (tidak terprediksi menderita penyakit paru-paru) dan "Ya" (terprediksi menderita penyakit paru-paru).
  
## Feedforward Neural Network architecture
![FNN](https://github.com/user-attachments/assets/29da4cf4-d78b-40fc-9fb4-854d81c96b6f)

## Random Forest architecture
![Random-forest-architecture-Breiman-2001](https://github.com/user-attachments/assets/3edef549-a281-4b08-a68c-5027e9f5ef74)

## Modelling
### Feedforward Neural Network
#### Preprocessing
Preprocessing data melibatkan beberapa langkah penting untuk menyiapkan dataset agar siap digunakan dalam model machine learning. Langkah pertama adalah pengecekan missing value, yang bertujuan untuk mengidentifikasi dan menangani data yang kosong. Setelah itu, dilakukan encoding label menggunakan LabelEncoder untuk mengubah data kategori menjadi numerik, sehingga model dapat memprosesnya. Selanjutnya, normalisasi dilakukan menggunakan MinMaxScaler, yang menskalakan data numerik ke rentang [0, 1], untuk menghindari bias akibat perbedaan skala antar fitur. Terakhir, data dibagi menjadi dua bagian, yaitu 80% untuk pelatihan dan 20% untuk pengujian.

#### Layer yang digunakan Feedforward Neural Network
![Screenshot 2024-12-21 091611](https://github.com/user-attachments/assets/b7e07de7-0dee-43ac-97ce-d78407748903)

#### Kurva
![image](https://github.com/user-attachments/assets/4f163d14-d907-4226-8aa4-2201661ea5b5)

#### Classification Report
![Screenshot 2024-12-21 092439](https://github.com/user-attachments/assets/4c8e6e1d-725d-4379-9647-efa24e0d2385)

#### Confusion Matrix
![Screenshot 2024-12-21 092551](https://github.com/user-attachments/assets/aac290f2-a49a-477d-be4b-9b75b46f082e)
Klasifikasi 0 : Tidak

Klasifikasi 1 : Ya

### Random Forest
#### Preprocessing
Untuk Preprocessing sama seperti yang digunakan model Feedforward Neural Network. Melibatkan beberapa langkah penting untuk menyiapkan dataset agar siap digunakan dalam model machine learning. Langkah pertama adalah pengecekan missing value, yang bertujuan untuk mengidentifikasi dan menangani data yang kosong. Setelah itu, dilakukan encoding label menggunakan LabelEncoder untuk mengubah data kategori menjadi numerik, sehingga model dapat memprosesnya. Selanjutnya, normalisasi dilakukan menggunakan MinMaxScaler, yang menskalakan data numerik ke rentang [0, 1], untuk menghindari bias akibat perbedaan skala antar fitur. Terakhir, data dibagi menjadi dua bagian, yaitu 80% untuk pelatihan dan 20% untuk pengujian.

#### Classification Report
![Screenshot 2024-12-21 093341](https://github.com/user-attachments/assets/a2d020f4-cb11-4984-bbae-83d36b778539)

#### Confusion Matrix
![Screenshot 2024-12-21 093453](https://github.com/user-attachments/assets/32cf96dc-b739-48e7-aa21-095acad05449)
Klasifikasi 0 : Tidak

Klasifikasi 1 : Ya








