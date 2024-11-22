import streamlit as st
import pandas as pd

def run():
    st.title("Upload Data")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Asli:")
        st.write(data.head())
        st.session_state['data'] = data  # Simpan data di session state
