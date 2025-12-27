import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cybersecurity Attack Classifier", page_icon="🛡️", layout="wide")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    try:
        # Using the EXACT filenames you uploaded (with the spaces)
        scaler = pickle.load(open('scaler .pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model .pkl', 'rb'))
        xgb_model = pickle.load(open('xgboost_model .pkl', 'rb'))
        lr_model = pickle.load(open('logistic_regression_model .pkl', 'rb'))
        target_encoder = pickle.load(open('label_encoder_target .pkl', 'rb'))
        
        with open('label_encoders.json', 'r') as f:
            feature_encoders = json.load(f)
            
        return scaler, rf_model, xgb_model, lr_model, target_encoder, feature_encoders
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e.filename}. Please ensure all .pkl and .json files are in the root folder.")
        return None

# --- UI LOGIC ---
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
        st.write("### Data Preview", df.head())
        
        if st.button("Run Detection"):
            try:
                # 1. Prepare data (Drop target/unnecessary columns)
                # Note: We match the columns used during your training
                X = df.copy()
                if 'Attack Type' in X.columns: X = X.drop(columns=['Attack Type'])
                if 'Unnamed: 15' in X.columns: X = X.drop(columns=['Unnamed: 15'])

                # 2. Categorical Encoding (Using the JSON encoders)
                for col in X.columns:
                    if col in feat_le_dict:
                        classes = feat_le_dict[col]['classes']
                        # Map known classes to integers, unknown to -1
                        mapping = {label: i for i, label in enumerate(classes)}
                        X[col] = X[col].map(mapping).fillna(-1)

                # 3. Scaling
                X_scaled = scaler.transform(X)
                
                # 4. Prediction
                preds = model.predict(X_scaled)
                decoded_labels = target_le.inverse_transform(preds)
                
                # 5. Display Results
                df['Prediction'] = decoded_labels
                st.success(f"Analysis complete using {model_name}!")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df[['ID', 'Title', 'Prediction']].head(20))
                with col2:
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x='Prediction', ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")
