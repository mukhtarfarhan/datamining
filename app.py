import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Setup page configuration
st.set_page_config(page_title="Data Science App", layout="wide", page_icon="📊")

# Simple login credentials (For demonstration purposes)
USER_CREDENTIALS = {"username": "admin", "password": "admin123"}

# Function to check if user is logged in
def check_user_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

# Function to show login form
def login_form():
    st.subheader("Login Required")
    username_input = st.text_input("Enter Username")
    password_input = st.text_input("Enter Password", type="password")
    
    if st.button("Login"):
        if username_input == USER_CREDENTIALS["username"] and password_input == USER_CREDENTIALS["password"]:
            st.session_state['logged_in'] = True
            st.success("Successfully logged in!")
        else:
            st.error("Invalid credentials, please try again.")

# Function for sidebar navigation
def display_sidebar():
    st.sidebar.header("Menu")
    choice = st.sidebar.radio("Select an Option:", ["Data Preparation", "Exploratory Data Analysis", "Model Training", "Prediction", "Cross-validation"])
    return choice

# Function to upload a dataset
def upload_data(key):
    uploaded_file = st.file_uploader(f"Upload Dataset for {key} (CSV)", type=["csv"], key=key)
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset successfully loaded!")
            return data
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    else:
        st.info(f"Please upload the dataset for {key}.")
        return None

# Function to handle missing values in the dataset
def process_missing_values(data):
    imputer = SimpleImputer(strategy="median")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Remove columns that are entirely empty
    empty_cols = [col for col in numerical_columns if data[col].isnull().all()]
    
    if empty_cols:
        st.warning(f"Removing fully empty columns: {empty_cols}")
        data = data.drop(columns=empty_cols)
        numerical_columns = [col for col in numerical_columns if col not in empty_cols]
    
    # Impute missing values
    imputed_data = imputer.fit_transform(data[numerical_columns])
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_columns, index=data.index)
    data[numerical_columns] = imputed_df
    
    return data

# Check user login status
check_user_login()

if not st.session_state['logged_in']:
    login_form()
else:
    # Main app content after successful login
    selected_option = display_sidebar()

    if selected_option == "Data Preparation":
        st.title("Data Preparation")
        st.subheader("Prepare and Clean Your Dataset")
        dataset = upload_data("Data Preparation")
        if dataset is not None:
            st.write("### Dataset Overview")
            st.dataframe(dataset.head())
            st.write("### Handle Missing Values")
            dataset = process_missing_values(dataset)
            st.dataframe(dataset.head())
            if st.button("Save Cleaned Dataset"):
                dataset.to_csv("Cleaned_Dataset.csv", index=False)
                st.success("Cleaned dataset saved successfully as 'Cleaned_Dataset.csv'.")

    elif selected_option == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        st.subheader("Analyze Your Data")
        dataset = upload_data("Exploratory Data Analysis")
        if dataset is not None:
            st.write("### Dataset Overview")
            st.dataframe(dataset.head())
            numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_columns) > 0:
                selected_column = st.selectbox("Choose a Column for Distribution Visualization:", numerical_columns)
                if selected_column:
                    st.write(f"### Distribution of {selected_column}")
                    fig = px.histogram(dataset, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig)
                if st.checkbox("Show Correlation Heatmap"):
                    if len(numerical_columns) > 1:
                        st.write("### Correlation Heatmap")
                        fig_corr, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(dataset[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig_corr)
                    else:
                        st.warning("Not enough numerical columns for correlation heatmap.")
            else:
                st.warning("No numerical columns available for analysis.")
            
            if st.checkbox("Show Descriptive Statistics"):
                st.write("### Descriptive Statistics")
                st.dataframe(dataset.describe())

            if st.checkbox("Show Scatter Matrix"):
                st.write("### Scatter Matrix")
                fig_scatter_matrix = px.scatter_matrix(
                    dataset, 
                    dimensions=dataset.select_dtypes(include=['float64', 'int64']).columns, 
                    title="Scatter Matrix of Features"
                )
                st.plotly_chart(fig_scatter_matrix)

    elif selected_option == "Model Training":
        st.title("Model Training")
        st.subheader("Train Your Machine Learning Model")
        dataset = upload_data("Model Training")
        if dataset is not None:
            st.write("### Dataset Overview")
            st.dataframe(dataset.head())
            target_column = st.selectbox("Select Target Variable:", dataset.columns)
            features = dataset.drop(columns=[target_column])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = dataset[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.success("Model trained successfully!")
            y_pred = model.predict(X_test)
            st.write("### Model Evaluation")
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            st.text(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
            
            if st.checkbox("Show Feature Importance"):
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                feature_importance = rf_model.feature_importances_
                feature_names = X.columns
                fig = px.bar(x=feature_names, y=feature_importance, title="Feature Importance")
                st.plotly_chart(fig)

    elif selected_option == "Prediction":
        st.title("Prediction")
        st.subheader("Make Predictions with New Data")
        training_data = upload_data("Prediction (Training)")
        if training_data is not None:
            st.write("### Training Dataset Overview")
            st.dataframe(training_data.head())
            target_column = st.selectbox("Select Target Variable (Training):", training_data.columns)
            features = training_data.drop(columns=[target_column])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = training_data[target_column]
            model = LinearRegression()
            model.fit(X, y)
            st.success("Model trained!")
            prediction_data = upload_data("Prediction (New Data)")
            if prediction_data is not None:
                st.write("### New Dataset Overview")
                st.dataframe(prediction_data.head())
                
                # Impute missing values
                imputed_pred_data = SimpleImputer(strategy="median").fit_transform(prediction_data.select_dtypes(include=['float64', 'int64']))

                # Check if number of columns match
                if imputed_pred_data.shape[1] == X.shape[1]:
                    imputed_pred_df = pd.DataFrame(imputed_pred_data, columns=X.columns, index=prediction_data.index)
                else:
                    st.error("Column mismatch between prediction data and training features.")
                    st.stop()

                # Predict
                predictions = model.predict(imputed_pred_df)
                st.write("### Prediction Results")
                pred_df = pd.DataFrame(predictions, columns=["Predictions"], index=prediction_data.index)
                st.dataframe(pred_df.head())

    elif selected_option == "Cross-validation":
        st.title("Cross-validation")
        st.subheader("Cross-validation for Model Evaluation")
        dataset = upload_data("Cross-validation")
        if dataset is not None:
            st.write("### Dataset Overview")
            st.dataframe(dataset.head())
            target_column = st.selectbox("Select Target Variable:", dataset.columns)
            features = dataset.drop(columns=[target_column])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = dataset[target_column]
            model = LinearRegression()
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
            st.write(f"### Cross-validation Scores: {scores}")
            st.write(f"### Average MSE: {np.mean(scores):.2f}")
