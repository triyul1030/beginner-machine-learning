# Beginner Machine Learning (Clustering & Klasifikasi)

## Deskripsi
Proyek ini merupakan implementasi algoritma **Machine Learning**, yaitu *Unsupervised Learning* (Clustering) dan *Supervised Learning* (Klasifikasi).

## Fitur dan Tahapan
1. **Exploratory Data Analysis (EDA)**
   - Menampilkan statistik dataset dengan fungsi `info()` dan `describe()`.
   - Menggali *insight* dari data menggunakan visualisasi matriks korelasi (`sns.heatmap`) dan distribusi tiap fitur (`sns.histplot`).
   - Menerapkan format visualisasi yang bersih dan tanpa teks *overlap* (`plt.xticks(rotation=45)`).

2. **Pembersihan dan Pra Pemrosesan Data (Preprocessing)**
   - Menangani tipe data *missing values* (`dropna()`) dan menghapus data duplikat (`drop_duplicates()`).
   - Melakukan efisiensi fitur dengan men-*drop* informasi personal/irrelevan (ID, Date, IP).
   - Menormalisasi data skala besar (Handling Outlier dengan IQR).
   - Mentransformasi fitur kategorikal (`LabelEncoder` & *One-Hot Encoding*) serta fitur numerik (`StandardScaler`).
   - Melakukan metode *binning* data berbasis kuantil numerik (`pd.qcut`).

3. **Membangun Model Clustering**
   - Memastikan dan menemukan jumlah klaster (K) yang paling optimal menggunakan **Elbow Method** dari *library* Yellowbrick (`KElbowVisualizer`).
   - Mengelompokkan data menggunakan algoritma **K-Means**.
   - Mengevaluasi kualitas *cluster* menggunakan perhitungan batas kemiripan objek **Silhouette Score**.
   - Mengaplikasikan teknik reduksi tingkat dimensi menggunakan **PCA (Principal Component Analysis)** untuk menyederhanakan data lalu memvisualisasikannya di ruang 2D.

4. **Interpretasi Hasil Clustering**
   - Mengekspor atribut hasil pemisahan *cluster* dan menamainya sebagai `Target`.
   - Melakukan *inverse_transform* untuk mengembalikan data yang telah di-*encode* dan di-*scale* ke nilai orisinalnya.
   - Melakukan analisis deskriptif (rata-rata, nilai min, max) untuk kolom numerik, serta analisis *modus* untuk kolom kategorikal.
   - Mengintegrasikan hasil penataan ulang *inverse* ini dalam file `data_clustering_inverse.csv` untuk keperluan lebih lanjut.

5. **Membangun Model Klasifikasi**
   - Memisahkan data evaluasi dari dataset `inverse` menggunakan fitur `train_test_split()`.
   - Membuat model peramal *cluster* dengan metode algoritma dasar **Decision Tree Classifier**.
   - Membandingkan performa model menggunakan algoritma kedua: **Random Forest Classifier**.
   - Melakukan uji perbaikan akurasi lanjutan (*Hyperparameter Tuning*) iteratif dengan algoritma **GridSearchCV**.
   - Menggunakan metrik `classification_report` untuk mengekstraksi informasi Precision, Recall, F-1 Score, dan Accuracy.

## File dan Aset
- `[Clustering]_Submission_Akhir_BMLP_Tri_Yulianto.ipynb`: File *Jupyter Notebook* terkait EDA hingga Clustering.
- `[Klasifikasi]_Submission_Akhir_BMLP_Tri_Yulianto.ipynb`: File *Jupyter Notebook* yang membangun peramal klaster dari algoritma Random Forest vs Decision Tree.
- `requirements.txt`: Kumpulan modul library.
- **Data (.csv)**: Kumpulan data awal maupun konversi inversi (seperti `data_clustering_inverse.csv`).
- **Models (.h5)**: 5 variasi kerangka kerja model AI (*export* menggunakan `joblib.dump()`) untuk *testing* / *autograder*.

## Cara Instalasi dan Penggunaan
1. Pastikan komputasi mendukung bahasa **Python 3.8+**.
2. Install *library* pihak ketiga melalui Command Prompt/Terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka *Jupyter Notebook / JupyterLab* atau VS Code:
   ```bash
   jupyter notebook
   ```
4. Buka file `[Clustering]_Submission_Akhir_BMLP_Tri_Yulianto.ipynb` lalu (`Run All`).
5. Lanjutkan *Run All* pada file klasifikasi.
