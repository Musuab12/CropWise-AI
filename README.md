# 🌾 CropWise AI

## Predictive AI for Optimal Crop Selection

CropWise AI is a predictive AI-based decision support system designed to help farmers select suitable crops based on soil and weather conditions.

### 🎯 Problem

Farmers often depend on traditional experience and general market trends when selecting crops. However, changing weather conditions, rainfall, temperature, soil characteristics, and water availability can make crop selection difficult.

### 💡 Our Solution

CropWise AI uses a **Random Forest Machine Learning model** to analyze agricultural conditions and recommend suitable crops.

The current prototype uses seven input features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The system provides the best crop recommendation along with the **Top 3 crop choices and their model scores**.

### 🤖 AI Technology

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib
- Streamlit

### 📊 Dataset & Experiment

The current prototype was trained and tested using a crop recommendation dataset containing:

- **2,200 records**
- **22 crop classes**
- **7 agricultural input features**

The Random Forest model achieved **99.55% test accuracy** in the current experiment.

> Note: This accuracy represents the test performance on the selected dataset. It is not Umarkot field-validation accuracy.

### 🖥️ Working Prototype

CropWise AI includes a Streamlit web application connected to the trained machine learning model.

Users can enter agricultural conditions and receive an AI-based crop recommendation.

### 📍 Research Focus

The initial research focus is **Umarkot, Sindh, Pakistan**.

Future development will integrate Umarkot-specific soil and weather data, real-time weather information, and additional AI techniques to improve localized recommendations and field validation.

### 🚀 Future Development

- Umarkot-specific agricultural datasets
- Real-time weather integration
- Soil texture and moisture information
- Salinity-related parameters
- Additional machine learning models
- Field validation with local agricultural data
- Farmer-friendly mobile/web deployment

### ⚠️ Responsible Use

CropWise AI is currently a research prototype. Its recommendations should be treated as decision-support information and should be validated with local agricultural knowledge and field conditions before making farming decisions.

### 👨‍💻 Project

**CropWise AI — Predictive AI for Optimal Crop Selection**

Developed for the **Alibaba Cloud AI Hackathon Pakistan 2026**.
