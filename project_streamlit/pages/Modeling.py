import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def show():
    st.title("Modeling")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca dataset
        df = pd.read_csv(uploaded_file)
        
        # Menampilkan nama kolom dataset
        st.write("Nama kolom dataset:", df.columns)
        
        # Menghapus kolom yang tidak relevan (misalnya kolom 'Unnamed')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Menghapus baris dengan nilai kosong
        df = df.dropna()

        # Reset index setelah pembersihan
        df = df.reset_index(drop=True)

        # Menampilkan data setelah pembersihan
        st.write("Data setelah pembersihan:", df.head())

        # Pilih target dan fitur
        target = st.selectbox("Pilih Kolom Target", df.columns)
        features = st.multiselect("Pilih Kolom Fitur", df.columns)

        # Menghapus kolom target dari fitur
        if target in features:
            features.remove(target)

        # Encoding kolom kategorikal jika ada
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].nunique() <= 10:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
        
        # Memastikan bahwa fitur dan target adalah numerik
        X = df[features]
        y = df[target]

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.text(classification_report(y_test, y_pred))
