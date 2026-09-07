
import streamlit as st
import pandas as pd
import joblib
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from datetime import date

# ============================================================
# CropWise AI — Pakistan Edition
# ============================================================

MODEL_FILE = "cropwise_rf_model.pkl"
model = joblib.load(MODEL_FILE)

st.set_page_config(
    page_title="CropWise AI | Pakistan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium UI styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f5faf7 0%, #ffffff 42%, #f7faf8 100%);
    }
    .block-container {
        max-width: 1250px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0b3d2e 0%, #126b4b 55%, #1b8a5a 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(20, 90, 65, .16);
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
        letter-spacing: -.5px;
    }
    .hero p {
        margin: .45rem 0 0;
        font-size: 1.08rem;
        opacity: .94;
    }
    .badge {
        display: inline-block;
        padding: .28rem .7rem;
        border-radius: 999px;
        background: rgba(255,255,255,.16);
        font-size: .82rem;
        margin-bottom: .75rem;
    }
    .section-note {
        color: #5f6d67;
        font-size: .9rem;
    }
    .result-card {
        padding: 1.35rem;
        border-radius: 20px;
        background: white;
        border: 1px solid #e4ece8;
        box-shadow: 0 8px 24px rgba(20, 70, 50, .07);
        text-align: center;
    }
    .result-card h3 { margin: 0; }
    .result-card .score {
        font-size: 1.65rem;
        font-weight: 700;
        margin: .35rem 0;
    }
    .info-card {
        padding: 1rem 1.15rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e7eeeb;
        min-height: 92px;
    }
    .info-card .label {
        color: #64736d;
        font-size: .82rem;
    }
    .info-card .value {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: .18rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4ece8;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 14px rgba(20, 70, 50, .04);
    }
    .footer {
        text-align: center;
        color: #718079;
        font-size: .82rem;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Research Location: Umarkot
# -----------------------------
UMARKOT_LAT = 25.3633
UMARKOT_LON = 69.7360
LOCATION_NAME = "Umarkot, Sindh, Pakistan"

SCENARIO_DATA = {'🌾 Rice': [90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362], '🌽 Maize': [71, 54, 16, 22.61359953, 63.69070564, 5.749914421, 87.75953857], '🫘 Chickpea': [40, 72, 77, 17.02498456, 16.98861173, 7.485996067, 88.55123143], '🌿 Cotton': [133, 47, 24, 24.40228894, 79.19732001, 7.231324765, 90.8022356], '🍌 Banana': [91, 94, 46, 29.36792366, 76.24900101, 6.149934034, 92.82840911], '🥭 Mango': [2, 40, 27, 29.73770045, 47.54885174, 5.954626604, 90.09586854], '🍎 Apple': [24, 128, 196, 22.75088787, 90.69489172, 5.521466996, 110.4317855]}

@st.cache_data(ttl=1800)
def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "CropWise-AI/2.0"})
    with urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))

@st.cache_data(ttl=86400)
def get_five_year_weather(lat, lon):
    end_year = date.today().year - 1
    start_year = end_year - 4
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "CropWise-AI/2.0"})
    with urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temperature": data["daily"]["temperature_2m_mean"],
        "rainfall": data["daily"]["precipitation_sum"],
    })
    df["year"] = df["date"].dt.year
    return df.groupby("year", as_index=False).agg(
        Avg_Temperature_C=("temperature", "mean"),
        Rainfall_mm=("rainfall", "sum"),
    )

def weather_description(code):
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain",
        65: "Heavy rain", 80: "Rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 95: "Thunderstorm",
    }
    return codes.get(code, "Weather condition available")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="badge">🇵🇰 Smart Agriculture • AI Research Prototype</div>
    <h1>🌾 CropWise AI</h1>
    <p>Predictive AI for Optimal Crop Selection — Umarkot, Sindh</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "A data-driven agricultural decision-support prototype combining "
    "machine learning with live and historical weather context."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 📍 Research Location")
st.sidebar.success("Umarkot, Sindh, Pakistan")
st.sidebar.caption(
    "Current research prototype is focused on Umarkot. "
    "Pakistan-wide expansion is planned after local validation."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Current AI Model")
st.sidebar.caption("Random Forest • 7 agricultural features")
st.sidebar.caption("Dataset: 2,200 records • 22 crop classes")
st.sidebar.caption("Test accuracy: 99.55%")

location_name = LOCATION_NAME
lat, lon = UMARKOT_LAT, UMARKOT_LON

# -----------------------------
# KPI strip
# -----------------------------
a, b, c, d = st.columns(4)
with a:
    st.metric("🤖 Model", "Random Forest")
with b:
    st.metric("🌾 Crop Classes", "22")
with c:
    st.metric("📚 Records", "2,200")
with d:
    st.metric("🎯 Test Accuracy", "99.55%")

st.divider()

# -----------------------------
# Weather
# -----------------------------
st.header(f"🌦️ Live Weather — {location_name}")
try:
    weather = get_weather(lat, lon)
    current = weather["current"]

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.metric("🌡️ Temperature", f'{current["temperature_2m"]:.1f} °C')
    with w2:
        st.metric("💧 Humidity", f'{current["relative_humidity_2m"]:.0f}%')
    with w3:
        st.metric("🌧️ Rain", f'{current["rain"]:.1f} mm')
    with w4:
        st.metric("☔ Precipitation", f'{current["precipitation"]:.1f} mm')

    st.caption(
        f'Condition: {weather_description(current["weather_code"])} • '
        f'Weather source: Open-Meteo • Updated: {current["time"]}'
    )

    with st.expander("📅 View 3-Day Weather Outlook"):
        daily = weather["daily"]
        forecast = pd.DataFrame({
            "Date": daily["time"],
            "Min Temp (°C)": daily["temperature_2m_min"],
            "Max Temp (°C)": daily["temperature_2m_max"],
            "Precipitation (mm)": daily["precipitation_sum"],
        })
        st.dataframe(forecast, use_container_width=True, hide_index=True)
except Exception:
    st.warning("Live weather is temporarily unavailable. Manual inputs remain available.")

# -----------------------------
# Five-year climate
# -----------------------------
st.header("📈 Five-Year Climate Context")
try:
    hist = get_five_year_weather(lat, lon)
    left, right = st.columns(2)
    with left:
        st.markdown("**Average Temperature by Year**")
        st.line_chart(hist.set_index("year")["Avg_Temperature_C"])
    with right:
        st.markdown("**Annual Rainfall by Year**")
        st.bar_chart(hist.set_index("year")["Rainfall_mm"])
    st.caption(
        "Historical climate context is based on weather/reanalysis data, "
        "not farmer-level field records."
    )
except Exception:
    st.info("Historical weather is temporarily unavailable.")

st.divider()

# -----------------------------
# Agricultural inputs
# -----------------------------
st.header("🌱 Soil & Agricultural Conditions")
st.markdown(
    '<div class="section-note">Enter soil/lab values when available. '
    'These seven fields match the current trained model.</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
    potassium = st.number_input("Potassium (K)", 0.0, 250.0, 50.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
with c2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

# Optional one-click sample
if st.button("✨ Use a Sample Condition Set"):
    st.info("Sample values are for demonstration only. Enter measured/local values for real analysis.")

st.divider()


# -----------------------------
# Demo scenarios
# -----------------------------
st.header("🎯 Quick Demo Scenarios")
st.markdown(
    '<div class="section-note">Use real sample conditions from the project dataset '
    'to demonstrate that the model responds to different agricultural profiles. '
    'These are demonstration samples, not farmer records.</div>',
    unsafe_allow_html=True,
)

SCENARIOS = SCENARIO_DATA
scenario = st.selectbox(
    "Choose a demonstration profile",
    ["Manual Input"] + list(SCENARIOS.keys()),
    index=0,
)

if scenario != "Manual Input":
    vals = SCENARIOS[scenario]
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = vals
    st.success(
        f"Loaded {scenario} sample conditions. "
        "Click **Predict Best Crop** below to run the AI model."
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
st.header("🤖 AI Crop Recommendation")

if st.button("🔍 Predict Best Crop", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    results = pd.DataFrame({
        "Crop": model.classes_,
        "Prediction Score": probabilities,
    }).sort_values("Prediction Score", ascending=False).head(3)

    st.success(f"🌾 Recommended Crop: **{prediction.title()}**")
    st.subheader("🏆 Top 3 Recommendations")

    cards = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(results.reset_index(drop=True).iterrows()):
        with cards[i]:
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{medals[i]} {row['Crop'].title()}</h3>
                    <div class="score">{row['Prediction Score']*100:.1f}%</div>
                    <div class="section-note">Prediction Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(row["Prediction Score"]))

    st.caption(
        "Prediction Score is the Random Forest probability estimate for the "
        "entered conditions. It is not a real-world probability of crop success."
    )

    st.subheader("🧠 AI Analysis")
    st.write(
        f"The model selected **{prediction.title()}** as the top recommendation "
        "for the entered agricultural conditions."
    )
    st.write(
        "The current model evaluates Nitrogen, Phosphorus, Potassium, "
        "temperature, humidity, soil pH and rainfall together."
    )

    with st.expander("📋 View Input Summary"):
        summary = pd.DataFrame({
            "Parameter": [
                "Nitrogen", "Phosphorus", "Potassium",
                "Temperature", "Humidity", "Soil pH", "Rainfall"
            ],
            "Value": [
                nitrogen, phosphorus, potassium,
                temperature, humidity, ph, rainfall
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# -----------------------------
# Data sources / roadmap
# -----------------------------
st.divider()
st.header("🗺️ Data & Future Integration")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("### 🌦️ Weather")
    st.write("Live and historical weather context.")
    st.link_button("Open Open-Meteo", "https://open-meteo.com/")
with s2:
    st.markdown("### 🌱 Soil")
    st.write("Global soil-data reference for future localized integration.")
    st.link_button("Open SoilGrids", "https://soilgrids.org/")
with s3:
    st.markdown("### 🌾 Agriculture")
    st.write("Umarkot and Sindh agricultural statistics for future localized analysis.")
    st.link_button("Open FAOSTAT", "https://www.fao.org/faostat/en/")

st.info(
    "Future development: integrate validated Pakistan-wide field soil data, "
    "farmer historical records, localized crop-yield data and additional AI "
    "models for location-aware crop suitability and yield prediction."
)

st.warning(
    "⚠️ Responsible Use: CropWise AI is a research prototype. The current "
    "Random Forest model is trained on the selected crop recommendation dataset "
    "and is not yet validated as a Umarkot field model."
)

st.markdown(
    '<div class="footer">CropWise AI • Alibaba Cloud AI Hackathon Pakistan 2026 • '
    'AI for Smart Agriculture</div>',
    unsafe_allow_html=True,
)
