import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json

# Ensure visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    st.error("Missing visualization libraries. Please check requirements.txt")

# --- CONFIGURATION ---
st.set_page_config(page_title="Cybersecurity Attack Classifier", page_icon="🛡️", layout="wide")

# --- ROBUST MODEL LOADING ---
@st.cache_resource
def load_resources():
    # Note: These strings match your uploaded filenames exactly (with the spaces)
    files = {
        "scaler": "scaler .pkl",
        "rf": "random_forest_model .pkl",
        "xgb": "xgboost_model .pkl",
        "lr": "logistic_regression_model .pkl",
        "target_le": "label_encoder_target .pkl",
        "feature_le": "label_encoders.json"
    }
    
    loaded = {}
    
    # Load Scaler
    try:
        with open(files["scaler"], 'rb') as f:
            loaded["scaler"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Scaler: {e}")
        return None

    # Load Models individually
    for key in ["rf", "xgb", "lr"]:
        try:
            with open(files[key], 'rb') as f:
                loaded[key] = pickle.load(f)
        except Exception as e:
            st.warning(f"Note: {key} model could not be loaded (File might be empty or version mismatch).")
            loaded[key] = None

    # Load Encoders
    try:
        with open(files["target_le"], 'rb') as f:
            loaded["target_le"] = pickle.load(f)
        with open(files["feature_le"], 'r') as f:
            loaded["feature_le"] = json.load(f)
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        return None
        
    return loaded

# --- UI LOGIC ---
st.title("🛡️ Cybersecurity Attack Classification")
data_bundle = load_resources()

if data_bundle:
    st.sidebar.header("Navigation")
    model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "XGBoost"])
    
    # Map selection to loaded models
    model_map = {"Random Forest": "rf", "Logistic Regression": "lr", "XGBoost": "xgb"}
    selected_model = data_bundle[model_map[model_option]]

    if selected_model is None:
        st.error(f"The {model_option} model is currently unavailable due to a file error.")
    else:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Raw Data Preview", df.head())
            
            if st.button("Run Detection"):
                try:
                    # Preprocessing
                    X = df.copy()
                    # Remove columns not used in training (adjust list as needed)
                    cols_to_drop = ['Attack Type', 'ID', 'Unnamed: 15']
                    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

                    # Apply Feature Encoding from JSON
                    for col, enc_data in data_bundle["feature_le"].items():
                        if col in X.columns:
                            mapping = {val: i for i, val in enumerate(enc_data['classes'])}
                            X[col] = X[col].map(mapping).fillna(-1)

                    # Scale and Predict
                    X_scaled = data_bundle["scaler"].transform(X)
                    preds = selected_model.predict(X_scaled)
                    labels = data_bundle["target_le"].inverse_transform(preds)
                    
                    # Display Result
                    df['Prediction'] = labels
                    st.success(f"Classification successful using {model_option}!")
                    st.dataframe(df[['Title', 'Prediction']] if 'Title' in df.columns else df)
                    
                    # Visualization
                    fig, ax = plt.subplots()
                    sns.countplot(x=labels, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
