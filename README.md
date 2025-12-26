# 🛡️ Multi-Class Classification of Cybersecurity Attacks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

A machine learning project for detecting and classifying cybersecurity attacks using network traffic data.

## 👨‍🎓 Project Information

- **Student:** Tayyab Ali
- **ID:** 2530-4007
- **Department:** Cyber Security
- **Course:** Artificial Intelligence & Machine Learning
- **Date:** December 24, 2025

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Deployment](#deployment)

## 🎯 Overview

This project implements a multi-class classification system to detect and classify various types of cybersecurity attacks using network traffic data. The system compares three machine learning algorithms and deploys the best model using Streamlit for real-time attack detection.

## ✨ Features

- **Multiple ML Algorithms**: Random Forest, Logistic Regression, and XGBoost
- **Advanced Preprocessing**: Handles missing values, outliers, and feature encoding
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Interactive Web App**: Streamlit deployment for real-time predictions
- **Visualizations**: Attack distribution, model comparison, feature importance
- **Model Persistence**: Trained models saved as .pkl files

## 📊 Dataset

**Source:** [Kaggle - Cybersecurity Attack and Defence Dataset](https://www.kaggle.com/datasets/tannubarot/cybersecurity-attack-and-defence-dataset)

**Attack Types Detected:**
- DDoS Attacks
- Port Scanning
- SQL Injection
- XSS (Cross-Site Scripting)
- Malware
- Benign Traffic
- And more...

### Dataset Download

```bash
kaggle datasets download -d tannubarot/cybersecurity-attack-and-defence-dataset
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/cybersecurity-attack-classification.git
cd cybersecurity-attack-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Training the Models

Run the main training script in Kaggle notebook to train models and save .pkl files.

### 2. Running the Streamlit App

Launch the web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Making Predictions

```python
import pickle
import pandas as pd

with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_target.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

data = pd.read_csv('your_data.csv')
predictions = model.predict(scaled_data)
predicted_labels = label_encoder.inverse_transform(predictions)
```

## 🤖 Models

### 1. Random Forest Classifier
- **Type:** Ensemble Learning
- **Parameters:** 100 estimators, random_state=42
- **Strengths:** Handles imbalanced data, robust to overfitting

### 2. Logistic Regression
- **Type:** Linear Model
- **Parameters:** max_iter=1000, random_state=42
- **Strengths:** Fast training, good baseline model

### 3. XGBoost Classifier
- **Type:** Gradient Boosting
- **Parameters:** 100 estimators, random_state=42
- **Strengths:** High accuracy, handles complex patterns

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.95 | 0.94 | 0.93 | 0.93 |
| Logistic Regression | 0.87 | 0.85 | 0.84 | 0.84 |
| XGBoost | **0.96** | **0.95** | **0.94** | **0.94** |

**Best Model:** XGBoost (F1-Score: 0.94)

## 📁 Project Structure

```
cybersecurity-attack-classification/
│
├── streamlit_app.py                      # Streamlit web application
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
│
└── models/                               # Trained model files
    ├── random_forest_model.pkl
    ├── logistic_regression_model.pkl
    ├── xgboost_model.pkl
    ├── scaler.pkl
    ├── label_encoder_target.pkl
    └── label_encoders.pkl
```

## 🌐 Deployment

### Local Deployment

```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

## 🎓 Academic Requirements

✅ **Data Collection** - Downloaded and processed Kaggle dataset  
✅ **Data Preprocessing** - Handled missing values, encoding, scaling  
✅ **Feature Engineering** - Feature selection and transformation  
✅ **Class Imbalance** - Applied SMOTE technique  
✅ **Model Training** - Trained 3 ML algorithms  
✅ **Evaluation Metrics** - Accuracy, Precision, Recall, F1-Score  
✅ **Visualizations** - Confusion matrix, feature importance  
✅ **Model Persistence** - Saved all models as .pkl files  
✅ **Deployment** - Created Streamlit web application  
✅ **Documentation** - Comprehensive README and code comments

## 📧 Contact

**Tayyab Ali**  
Department of Cyber Security  
Email: tayyabali@example.com

**Project Link:** [https://github.com/yourusername/cybersecurity-attack-classification](https://github.com/yourusername/cybersecurity-attack-classification)

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset
- [Tannu Barot](https://www.kaggle.com/tannubarot) for uploading the dataset
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [Streamlit](https://streamlit.io/) for web deployment framework

---

⭐ **If you find this project helpful, please give it a star!** ⭐
