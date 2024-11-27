import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def show():
    st.title("Prediction")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Membaca dataset
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
        # Pilih kolom target dan fitur
        target = st.selectbox("Pilih Kolom Target", df.columns)
        features = st.multiselect("Pilih Kolom Fitur", df.columns)
        
        if len(features) > 0:
            X = df[features]
            y = df[target]
            
            # Latih model
            model = RandomForestClassifier()
            model.fit(X, y)
            
            # Input untuk prediksi
            input_data = []
            for feature in features:
                value = st.number_input(f"Masukkan nilai untuk {feature}", value=0)
                input_data.append(value)
            
            if st.button("Prediksi"):
                prediction = model.predict([input_data])
                st.write(f"Prediksi: {prediction[0]}")
