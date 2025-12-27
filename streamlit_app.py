import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json

# Try-Except blocks for imports to help debug in Streamlit Cloud
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    st.error("Missing libraries! Please ensure 'matplotlib' and 'seaborn' are in your requirements.txt")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cybersecurity Attack Classifier", page_icon="🛡️", layout="wide")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    try:
        # NOTE: Using the exact filenames from your upload (with the spaces)
        scaler = pickle.load(open('scaler .pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model .pkl', 'rb'))
        xgb_model = pickle.load(open('xgboost_model .pkl', 'rb'))
        lr_model = pickle.load(open('logistic_regression_model .pkl', 'rb'))
        target_encoder = pickle.load(open('label_encoder_target .pkl', 'rb'))
        
        with open('label_encoders.json', 'r') as f:
            feature_encoders = json.load(f)
            
        return scaler, rf_model, xgb_model, lr_model, target_encoder, feature_encoders
    except FileNotFoundError as e:
        st.error(f"❌ File missing from GitHub: {e.filename}")
        st.info("Ensure files like 'scaler .pkl' are uploaded to your repository.")
        return None

# --- UI ---
st.title("🛡️ Cybersecurity Attack Classification")

res = load_resources()

if res:
    scaler, rf, xgb, lr, target_le, feat_le_dict = res
    
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Logistic Regression"])
    model = {"Random Forest": rf, "XGBoost": xgb, "Logistic Regression": lr}[model_name]

    uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Input Data Preview", df.head())
        
        if st.button("Predict"):
            try:
                # Preprocessing
                X = df.copy()
                # Drop non-feature columns if they exist
                for col in ['Attack Type', 'Unnamed: 15', 'ID']:
                    if col in X.columns: X = X.drop(columns=[col])

                # Scaling
                X_scaled = scaler.transform(X)
                
                # Predict
                preds = model.predict(X_scaled)
                decoded = target_le.inverse_transform(preds)
                
                df['Prediction'] = decoded
                st.success("Analysis Complete!")
                st.dataframe(df[['Title', 'Prediction']].head(10))
                
                # Visualization
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='Prediction', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
