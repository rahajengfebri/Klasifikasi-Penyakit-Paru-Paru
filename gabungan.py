import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model FFNN dan Random Forest
ffnn_model = load_model('Code/ffnn_model.h5')
rf_model = pickle.load(open('Code/random_forest_model (4).sav', 'rb'))

# Mapping untuk konversi string ke numerik
usia_map = {'Muda': 0, 'Tua': 1}
jenis_kelamin_map = {'Pria': 0, 'Wanita': 1}
merokok_map = {'Aktif': 0, 'Pasif': 1}
bekerja_map = {'Tidak': 0, 'Ya': 1}
rumah_tangga_map = {'Tidak': 0, 'Ya': 1}
begadang_map = {'Tidak': 0, 'Ya': 1}
olahraga_map = {'Jarang': 0, 'Sering': 1}
asuransi_map = {'Ada': 0, 'Tidak': 1}
penyakit_map = {'Ada': 0, 'Tidak': 1}

# Judul aplikasi
st.title("Prediksi Penyakit Paru-Paru")

# Input pengguna dalam 3 kolom
st.subheader("Masukkan Data Pasien")
col1, col2, col3 = st.columns(3)

with col1:
    usia = st.selectbox("Usia", list(usia_map.keys()))
    jenis_kelamin = st.selectbox("Jenis Kelamin", list(jenis_kelamin_map.keys()))
    merokok = st.selectbox("Kebiasaan Merokok", list(merokok_map.keys()))

with col2:
    bekerja = st.selectbox("Apakah Bekerja?", list(bekerja_map.keys()))
    rumah_tangga = st.selectbox("Rumah Tangga", list(rumah_tangga_map.keys()))
    begadang = st.selectbox("Kebiasaan Begadang", list(begadang_map.keys()))

with col3:
    olahraga = st.selectbox("Frekuensi Olahraga", list(olahraga_map.keys()))
    asuransi = st.selectbox("Punya Asuransi Kesehatan?", list(asuransi_map.keys()))
    penyakit = st.selectbox("Ada Riwayat Penyakit?", list(penyakit_map.keys()))

# Pilih metode prediksi
st.subheader("Pilih Metode Prediksi")
model_option = st.selectbox("Metode", ["Feedforward neural network", "Random Forest"])

# Tombol prediksi
if st.button("Prediksi"):
    # Konversi input ke numerik
    usia_num = usia_map[usia]
    jenis_kelamin_num = jenis_kelamin_map[jenis_kelamin]
    merokok_num = merokok_map[merokok]
    bekerja_num = bekerja_map[bekerja]
    rumah_tangga_num = rumah_tangga_map[rumah_tangga]
    begadang_num = begadang_map[begadang]
    olahraga_num = olahraga_map[olahraga]
    asuransi_num = asuransi_map[asuransi]
    penyakit_num = penyakit_map[penyakit]

    # Buat input data dalam format array
    input_data = np.array([[usia_num, jenis_kelamin_num, merokok_num, bekerja_num, rumah_tangga_num,
                            begadang_num, olahraga_num, asuransi_num, penyakit_num]])

    if model_option == "Feedforward neural network":
        # Prediksi dengan model FNN
        prediction = ffnn_model.predict(input_data)
        probability = prediction[0][0]

    elif model_option == "Random Forest":
        # Prediksi dengan model Random Forest
        probability = rf_model.predict_proba(input_data)[0][1]

    # Menentukan hasil berdasarkan threshold 0.5
    if probability > 0.5:
        st.error(f"Probabilitas: {probability:.2f} - Terdeteksi memiliki penyakit paru-paru.")
    else:
        st.success(f"Probabilitas: {probability:.2f} - Tidak terdeteksi memiliki penyakit paru-paru.")
