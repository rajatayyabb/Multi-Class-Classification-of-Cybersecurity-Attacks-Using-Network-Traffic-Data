import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Cybersecurity Attack Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; font-weight: bold; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_file(filename):
    """Helper to load pickle files with error handling for the specific filenames uploaded."""
    try:
        # Using exact filenames as they appear in your uploaded list
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: File '{filename}' not found. Please ensure it is in the project root.")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

@st.cache_resource
def load_all_resources():
    # Note: These names match your uploaded file list exactly (including trailing spaces)
    models = {
        'Random Forest': load_model_file('random_forest_model .pkl'),
        'Logistic Regression': load_model_file('logistic_regression_model .pkl'),
        'XGBoost': load_model_file('xgboost_model .pkl')
    }
    
    scaler = load_model_file('scaler .pkl')
    label_encoder_target = load_model_file('label_encoder_target .pkl')
    
    # Load label encoders for features from JSON
    label_encoders_features = None
    try:
        if os.path.exists('label_encoders.json'):
            with open('label_encoders.json', 'r') as f:
                le_dict = json.load(f)
            
            label_encoders_features = {}
            for col, data in le_dict.items():
                le = LabelEncoder()
                le.classes_ = np.array(data['classes'])
                label_encoders_features[col] = le
    except Exception as e:
        st.warning(f"Could not load feature encoders: {e}")
    
    return models, scaler, label_encoder_target, label_encoders_features

def show_home():
    st.header("Welcome to the Cybersecurity Attack Classification System")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("This project implements a multi-class classification system to detect and classify various types of cybersecurity attacks using network traffic data.")
    with col2:
        st.subheader("📊 Dataset Information")
        st.write("Source: Kaggle - Cybersecurity Attack Dataset. Detects: DDoS, Port Scanning, SQL Injection, XSS, Malware, and Benign traffic.")

def show_prediction(models, scaler, label_encoder_target, label_encoders_features):
    st.header("🔍 Attack Prediction")
    selected_model_name = st.selectbox("Choose a classification model:", list(models.keys()))
    model = models[selected_model_name]
    
    if model is None or scaler is None:
        st.error("Resources missing. Check your .pkl files.")
        return

    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded!")
            
            # Prepare data (Drop target if exists)
            target_cols = ['Attack Type', 'attack_type', 'label', 'Label']
            X = df.drop(columns=[c for c in target_cols if c in df.columns])
            
            # Handle categorical features using the loaded JSON encoders
            if label_encoders_features:
                for col in X.select_dtypes(include=['object']).columns:
                    if col in label_encoders_features:
                        le = label_encoders_features[col]
                        X[col] = X[col].map(lambda s: le.transform([str(s)])[0] if str(s) in le.classes_ else -1)

            # Scale and Predict
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            
            # Decode predictions
            decoded_preds = label_encoder_target.inverse_transform(preds)
            df['Predicted_Attack'] = decoded_preds
            
            st.subheader("Results")
            st.dataframe(df.head(20))
            
            # Visualization
            fig, ax = plt.subplots()
            sns.countplot(x=decoded_preds, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

def main():
    st.markdown('<p class="main-header">🛡️ Cybersecurity Attack Classifier</p>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Prediction"])
    
    resources = load_all_resources()
    
    if page == "🏠 Home":
        show_home()
    else:
        show_prediction(*resources)

if __name__ == "__main__":
    main()
