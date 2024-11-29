import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("Exploratory Data Analysis (EDA)")
    
    # Upload file CSV
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Membaca dan menampilkan dataset
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
        # Tampilkan statistik deskriptif
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
        
        # Visualisasi distribusi kolom numerik
        st.subheader("Distribusi Kolom")
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax, kde=True)
            st.pyplot(fig)
