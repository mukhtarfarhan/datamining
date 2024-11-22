import streamlit as st
import pandas as pd

def run():
    st.title("Data Preprocessing")

    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.write("Silakan unggah dataset terlebih dahulu melalui halaman 'Upload Data'.")
        return

    data = st.session_state['data']
    st.write("Data Sebelum Preprocessing:")
    st.write(data.head())

    st.write("**Preprocessing**")
    if st.button("Lakukan Preprocessing"):
        # Konversi semua kolom menjadi numerik, mengganti nilai non-numerik dengan NaN
        data = data.apply(pd.to_numeric, errors='coerce')
        data.fillna(data.mean(), inplace=True)
        st.write("Data Setelah Preprocessing:")
        st.write(data.head())
        st.session_state['data'] = data

        # Memeriksa nilai yang hilang
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            st.write(f"Masih ada {total_missing} nilai yang hilang setelah preprocessing.")
            st.write("Kolom dengan nilai yang hilang:")
            for col, num_missing in missing_values.items():
                if num_missing > 0:
                    st.write(f"{col}: {num_missing} nilai yang hilang")
        else:
            st.write("Tidak ada nilai yang hilang setelah preprocessing.")
