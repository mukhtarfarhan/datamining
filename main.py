import streamlit as st
from pages import home, data_upload, data_preprocessing, eda, model_evaluation, model_prediction

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Home", "Upload Data", "Data Preprocessing", "EDA", "Model Evaluation", "Model Prediction"])

if page == "Home":
    home.run()
elif page == "Upload Data":
    data_upload.run()
elif page == "Data Preprocessing":
    data_preprocessing.run()
elif page == "EDA":
    eda.run()
elif page == "Model Evaluation":
    model_evaluation.run()
elif page == "Model Prediction":
    model_prediction.run()
