import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    st.title("Exploratory Data Analysis (EDA)")

    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.write("Silakan unggah dataset terlebih dahulu melalui halaman 'Upload Data'.")
        return

    data = st.session_state['data']
    st.write("Data yang Dimuat:")
    st.write(data.head())

    # Menangani nilai yang hilang
    st.write("**Statistik Deskriptif**")
    st.write(data.describe())

    st.write("**Visualisasi Interaktif**")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Filter out columns with missing values
    numerical_columns = [col for col in numerical_columns if data[col].notnull().all()]

    if not numerical_columns:
        st.write("Tidak ada kolom numerik yang lengkap untuk divisualisasikan.")
    else:
        for column in numerical_columns:
            if data[column].isnull().sum() == 0:  # Pastikan tidak ada nilai hilang
                fig = px.histogram(data, x=column, title=f'Distribusi {column}')
                st.plotly_chart(fig)
            else:
                st.write(f"Kolom {column} memiliki nilai yang hilang dan tidak akan divisualisasikan.")
