import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def run():
    st.title("Prediksi dengan Model")

    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.write("Silakan unggah dataset terlebih dahulu melalui halaman 'Upload Data'.")
        return

    data = st.session_state['data']
    st.write("Data yang Dimuat:")
    st.write(data.head())

    # Menampilkan nama kolom untuk identifikasi
    st.write("Nama Kolom dalam Dataset:")
    columns = list(data.columns)
    st.write(columns)

    # Pilih kolom target menggunakan selectbox
    target_column = st.selectbox("Pilih kolom target:", options=columns)

    if target_column:
        # Preprocessing data
        X = data.drop(columns=target_column).apply(pd.to_numeric, errors='coerce').fillna(0)
        y = data[target_column].apply(pd.to_numeric, errors='coerce').fillna(0)
        st.write("Kolom features dan target berhasil dipisahkan.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("Data berhasil dibagi menjadi training dan testing.")

        # Load the best model
        if 'best_model' in st.session_state and 'feature_names' in st.session_state:
            best_model = st.session_state['best_model']
            best_model_name = st.session_state['best_model_name']
            feature_names = st.session_state['feature_names']
            
            # Pastikan data prediksi memiliki fitur yang sama dengan data pelatihan
            X_test = X_test[feature_names]
            
            st.write(f"Menggunakan model terbaik: {best_model_name}")
            
            # Prediksi dan evaluasi
            try:
                y_pred = best_model.predict(X_test)
                st.write("**Laporan Klasifikasi:**")
                st.text(classification_report(y_test, y_pred))
                st.write("**Akurasi:**", accuracy_score(y_test, y_pred))
            except Exception as e:
                st.write(f"Kesalahan saat membuat prediksi: {e}")
        else:
            st.write("Model terbaik belum dievaluasi atau nama fitur tidak tersedia. Silakan lakukan evaluasi model terlebih dahulu.")
    else:
        st.write("Pilih kolom target untuk melanjutkan.")
