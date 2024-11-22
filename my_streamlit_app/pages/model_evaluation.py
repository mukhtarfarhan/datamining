import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px

def run():
    st.title("Evaluasi Model")

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
        # Konversi kolom target menjadi kategori diskret
        y = data[target_column].apply(pd.to_numeric, errors='coerce').fillna(0)
        bins = st.slider("Pilih jumlah kategori:", 2, 10, 3)
        y_binned = pd.cut(y, bins, labels=False)

        # Preprocessing data
        X = data.drop(columns=target_column).apply(pd.to_numeric, errors='coerce').fillna(0)
        st.write("Kolom features dan target berhasil dipisahkan.")

        # Tampilkan distribusi kolom target
        st.write(f"Distribusi {target_column}")
        fig = px.histogram(data, x=target_column, title=f'Distribusi {target_column}')
        st.plotly_chart(fig)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)
        st.write("Data berhasil dibagi menjadi training dan testing.")
        
        # Simpan nama fitur yang digunakan saat pelatihan
        st.session_state['feature_names'] = X_train.columns.tolist()
        
        # List of models
        models = {
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Support Vector Machine': SVC()
        }
        
        # Evaluate models
        st.write("**Hasil Evaluasi Model**")
        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(f"### {model_name}")
            st.write(f"**Akurasi:** {accuracy:.2f}")
            
            # Pastikan panjang x dan y sama sebelum membuat grafik
            report_keys = list(report.keys())
            f1_scores = [report[cls]['f1-score'] for cls in report if cls not in ['accuracy', 'macro avg', 'weighted avg']]
            if len(report_keys) == len(f1_scores):
                fig = px.bar(x=report_keys, y=f1_scores, title=f'F1-Score per Kelas untuk {model_name}')
                st.plotly_chart(fig)
            else:
                st.write(f"Panjang x dan y tidak cocok untuk {model_name}.")
            
            results[model_name] = accuracy
        
        # Benchmarking
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        st.write(f"**Model Terbaik:** {best_model_name} dengan akurasi {results[best_model_name]:.2f}")
        
        # Save the best model
        st.session_state['best_model'] = best_model
        st.session_state['best_model_name'] = best_model_name
    else:
        st.write("Pilih kolom target untuk melanjutkan.")
