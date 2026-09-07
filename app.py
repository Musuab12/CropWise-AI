import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("cropwise_rf_model.pkl")

# Page configuration
st.set_page_config(
    page_title="CropWise AI",
    page_icon="🌾",
    layout="wide"
)

# Header
st.title("🌾 CropWise AI")
st.subheader("Predictive AI for Optimal Crop Selection")

st.write(
    "An AI-based decision support system that recommends suitable "
    "crops using soil and weather conditions."
)

st.info(
    "📍 Prototype Focus: Umarkot, Sindh, Pakistan"
)

st.divider()

# Input section
st.header("🌱 Agricultural Conditions")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)", 0.0, 200.0, 50.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)", 0.0, 200.0, 50.0
    )

    potassium = st.number_input(
        "Potassium (K)", 0.0, 250.0, 50.0
    )

    temperature = st.number_input(
        "Temperature (°C)", 0.0, 50.0, 25.0
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, 70.0
    )

    ph = st.number_input(
        "Soil pH", 0.0, 14.0, 6.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 500.0, 100.0
    )

st.divider()

# Prediction
if st.button("🔍 Predict Best Crop", use_container_width=True):

    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    results = pd.DataFrame({
        "Crop": model.classes_,
        "Suitability": probabilities
    })

    results = results.sort_values(
        "Suitability",
        ascending=False
    ).head(3)

    # Main result
    st.success(
        f"🌾 Recommended Crop: **{prediction.title()}**"
    )

    # Top recommendations
    st.header("🏆 Top 3 Crop Recommendations")

    for i, row in results.reset_index(drop=True).iterrows():

        score = row["Suitability"]

        st.write(
            f"**{i + 1}. {row['Crop'].title()}** — "
            f"{score * 100:.2f}% model score"
        )

        st.progress(float(score))

    st.divider()

    # AI analysis
    st.header("🤖 AI Analysis")

    st.write(
        f"Based on the entered conditions, the Random Forest model "
        f"selected **{prediction.title()}** as the top recommendation."
    )

    st.write(
        "The model considers nitrogen, phosphorus, potassium, "
        "temperature, humidity, soil pH and rainfall together "
        "to generate the recommendation."
    )

    # Input summary
    st.header("📊 Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "Soil pH",
            "Rainfall"
        ],
        "Value": [
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Note: This is an AI research prototype. "
        "The current model is trained on the selected crop recommendation dataset "
        "and is not yet a validated Umarkot field model."
    )

st.divider()

st.caption(
    "CropWise AI | Predictive AI for Optimal Crop Selection"
)
