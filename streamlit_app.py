import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Try to import visualization libraries, handle if missing
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    st.error("Visualization libraries missing. Please update requirements.txt")

# --- CONFIG ---
st.set_page_config(page_title="Cybersecurity Classifier", page_icon="🛡️", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_all_assets():
    # Exact filenames as they appear in your uploaded list (with spaces)
    file_map = {
        "scaler": "scaler .pkl",
        "rf": "random_forest_model .pkl",
        "xgb": "xgboost_model .pkl",
        "lr": "logistic_regression_model .pkl",
        "target_le": "label_encoder_target .pkl",
        "feat_le": "label_encoders.json"
    }
    
    assets = {}
    
    # Load Scaler and Encoders (Required)
    try:
        with open(file_map["scaler"], 'rb') as f:
            assets["scaler"] = pickle.load(f)
        with open(file_map["target_le"], 'rb') as f:
            assets["target_le"] = pickle.load(f)
        with open(file_map["feat_le"], 'r') as f:
            assets["feat_le"] = json.load(f)
    except Exception as e:
        st.error(f"Critical Error loading base assets: {e}")
        return None

    # Load Models (If one fails, the others can still work)
    for key in ["rf", "xgb", "lr"]:
        try:
            with open(file_map[key], 'rb') as f:
                assets[key] = pickle.load(f)
        except Exception:
            assets[key] = None # Mark as unavailable
            
    return assets

# --- APP UI ---
st.title("🛡️ Cybersecurity Attack Classification")
data = load_all_assets()

if data:
    st.sidebar.header("Model Selection")
    choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "XGBoost"])
    
    # Selection mapping
    model_key = {"Random Forest": "rf", "Logistic Regression": "lr", "XGBoost": "xgb"}[choice]
    model = data[model_key]

    if model is None:
        st.warning(f"The {choice} model is corrupted or missing. Try another.")
    else:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview", df.head())
            
            if st.button("Run Prediction"):
                try:
                    # 1. Feature Prep
                    X = df.copy()
                    drop_cols = ['Attack Type', 'ID', 'Unnamed: 15']
                    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

                    # 2. Encode categorical columns using JSON data
                    for col, enc in data["feat_le"].items():
                        if col in X.columns:
                            mapping = {val: i for i, val in enumerate(enc['classes'])}
                            X[col] = X[col].map(mapping).fillna(-1)

                    # 3. Scale and Predict
                    X_scaled = data["scaler"].transform(X)
                    preds = model.predict(X_scaled)
                    labels = data["target_le"].inverse_transform(preds)
                    
                    # 4. Show Results
                    df['Prediction'] = labels
                    st.success("Analysis Complete!")
                    st.dataframe(df[['Title', 'Prediction']] if 'Title' in df.columns else df)
                    
                    fig, ax = plt.subplots()
                    sns.countplot(x=labels, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction Failed: {e}")
