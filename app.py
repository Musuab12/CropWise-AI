import streamlit as st
import pandas as pd
import joblib
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("cropwise_rf_model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CropWise AI",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# Umarkot location
# -----------------------------
UMARKOT_LAT = 25.3655
UMARKOT_LON = 69.7401


@st.cache_data(ttl=1800)
def get_live_weather():
    """Fetch current weather and a short forecast for Umarkot."""
    params = {
        "latitude": UMARKOT_LAT,
        "longitude": UMARKOT_LON,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "Asia/Karachi"
    }

    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "CropWise-AI/1.0"})

    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_description(code):
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm"
    }
    return codes.get(code, "Weather condition available")


# -----------------------------
# Header
# -----------------------------
st.title("🌾 CropWise AI")
st.subheader("Predictive AI for Optimal Crop Selection")

st.write(
    "An AI-based decision support system that recommends suitable "
    "crops using soil and weather conditions."
)

st.info("📍 Prototype Focus: Umarkot, Sindh, Pakistan")

# -----------------------------
# Live Weather
# -----------------------------
st.header("🌦️ Live Weather — Umarkot")

weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

try:
    weather = get_live_weather()
    current = weather["current"]

    with weather_col1:
        st.metric("Temperature", f'{current["temperature_2m"]:.1f} °C')

    with weather_col2:
        st.metric("Humidity", f'{current["relative_humidity_2m"]:.0f} %')

    with weather_col3:
        st.metric("Rain", f'{current["rain"]:.1f} mm')

    with weather_col4:
        st.metric("Precipitation", f'{current["precipitation"]:.1f} mm')

    st.caption(
        f'Current condition: {weather_description(current["weather_code"])} '
        f'| Weather data: Open-Meteo | Updated: {current["time"]}'
    )

    with st.expander("📅 3-Day Weather Outlook"):
        daily = weather["daily"]
        forecast = pd.DataFrame({
            "Date": daily["time"],
            "Min Temperature (°C)": daily["temperature_2m_min"],
            "Max Temperature (°C)": daily["temperature_2m_max"],
            "Precipitation (mm)": daily["precipitation_sum"]
        })
        st.dataframe(forecast, use_container_width=True, hide_index=True)

except Exception:
    st.warning(
        "Live weather is temporarily unavailable. "
        "You can still use the manual agricultural inputs below."
    )

st.divider()

# -----------------------------
# Agricultural Conditions
# -----------------------------
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

st.caption(
    "The current Random Forest model uses seven dataset-compatible inputs: "
    "N, P, K, temperature, humidity, pH and rainfall."
)

# -----------------------------
# Soil Data Source
# -----------------------------
st.header("🌱 Soil Data Integration")

st.info(
    "SoilGrids provides global soil-property maps including pH and total nitrogen. "
    "The current prototype keeps N, P, K and pH as user inputs so the prediction "
    "remains compatible with the trained model."
)

st.link_button(
    "🌍 Explore SoilGrids Data",
    "https://soilgrids.org/"
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
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

    prediction = model.predict(input_data)[0]
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

    st.header("🏆 Top 3 Crop Recommendations")

    for i, row in results.reset_index(drop=True).iterrows():

        score = row["Suitability"]

        st.write(
            f"**{i + 1}. {row['Crop'].title()}** — "
            f"{score * 100:.2f}% prediction score"
        )

        st.progress(float(score))

    st.caption(
        "Prediction score is the Random Forest probability estimate for the "
        "entered conditions; it is not a real-world crop success probability."
    )

    st.divider()

    # AI analysis
    st.header("🤖 AI Analysis")

    st.write(
        f"Based on the entered conditions, the Random Forest model "
        f"selected **{prediction.title()}** as the top recommendation."
    )

    st.write(
        "The model considers nitrogen, phosphorus, potassium, temperature, "
        "humidity, soil pH and rainfall together to generate the recommendation."
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
    "CropWise AI | Predictive AI for Optimal Crop Selection | "
    "Weather data: Open-Meteo | Soil reference: SoilGrids"
)
