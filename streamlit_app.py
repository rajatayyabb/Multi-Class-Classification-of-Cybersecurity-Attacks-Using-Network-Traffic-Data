import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    # Exact filenames based on your upload (with spaces)
    try:
        scaler = pickle.load(open('scaler .pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model .pkl', 'rb'))
        xgb_model = pickle.load(open('xgboost_model .pkl', 'rb'))
        lr_model = pickle.load(open('logistic_regression_model .pkl', 'rb'))
        target_encoder = pickle.load(open('label_encoder_target .pkl', 'rb'))
        
        # Load JSON feature encoders
        with open('label_encoders.json', 'r') as f:
            feature_encoders = json.load(f)
            
        return scaler, rf_model, xgb_model, lr_model, target_encoder, feature_encoders
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e.filename}. Please check your file names in the repository.")
        return None

# --- UI SETUP ---
st.title("🛡️ Cybersecurity Attack Classifier")

resources = load_resources()

if resources:
    scaler, rf, xgb, lr, target_le, feat_le = resources
    st.success("✅ Models and Scaler loaded successfully!")
    
    # Model Selection
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Logistic Regression"])
    selected_model = {"Random Forest": rf, "XGBoost": xgb, "Logistic Regression": lr}[model_choice]

    # File Upload for Prediction
    uploaded_file = st.file_uploader("Upload Network Traffic Data (CSV)", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Input Data Preview:", data.head())
        
        if st.button("Predict Attacks"):
            # Ensure we only use columns the model was trained on
            # This step depends on your specific dataset features
            try:
                # 1. Feature Encoding (using the JSON mapping)
                # 2. Scaling
                scaled_data = scaler.transform(data)
                # 3. Prediction
                preds = selected_model.predict(scaled_data)
                labels = target_le.inverse_transform(preds)
                
                data['Prediction'] = labels
                st.write("### Prediction Results")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
