
import streamlit as st
import pandas as pd
import joblib
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from datetime import date

# -----------------------------
# Model
# -----------------------------
model = joblib.load("cropwise_rf_model.pkl")

st.set_page_config(
    page_title="CropWise AI | Pakistan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main { background-color: #f7faf8; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f5132, #198754);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p { margin: .35rem 0 0; font-size: 1.05rem; }
    .small-note { color: #5f6b66; font-size: .88rem; }
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 10px;
        border: 1px solid #e6ece8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Pakistan location
# -----------------------------
PAKISTAN = {
    "Umarkot, Sindh": (25.3633, 69.7360),
    "Hyderabad, Sindh": (25.3960, 68.3578),
    "Karachi, Sindh": (24.8607, 67.0011),
    "Sukkur, Sindh": (27.7244, 68.8228),
    "Larkana, Sindh": (27.5600, 68.2264),
    "Nawabshah, Sindh": (26.2442, 68.4100),
    "Mirpur Khas, Sindh": (25.5276, 69.0111),
    "Thatta, Sindh": (24.7461, 67.9236),
    "Multan, Punjab": (30.1575, 71.5249),
    "Lahore, Punjab": (31.5204, 74.3587),
    "Faisalabad, Punjab": (31.4504, 73.1350),
    "Bahawalpur, Punjab": (29.3956, 71.6836),
    "Sahiwal, Punjab": (30.6682, 73.1114),
    "Rahim Yar Khan, Punjab": (28.4212, 70.2989),
    "Dera Ghazi Khan, Punjab": (30.0561, 70.6348),
    "Gujranwala, Punjab": (32.1877, 74.1945),
    "Peshawar, Khyber Pakhtunkhwa": (34.0151, 71.5249),
    "Dera Ismail Khan, Khyber Pakhtunkhwa": (31.8311, 70.9017),
    "Mardan, Khyber Pakhtunkhwa": (34.1980, 72.0400),
    "Swat, Khyber Pakhtunkhwa": (35.2227, 72.4258),
    "Quetta, Balochistan": (30.1798, 66.9750),
    "Sibi, Balochistan": (29.5448, 67.8773),
    "Turbat, Balochistan": (26.0023, 63.0505),
    "Gwadar, Balochistan": (25.1264, 62.3225),
    "Islamabad, ICT": (33.6844, 73.0479),
    "Muzaffarabad, AJK": (34.3700, 73.4711),
    "Gilgit, Gilgit-Baltistan": (35.9208, 74.3089),
}

@st.cache_data(ttl=3600)
def geocode_pakistan(place):
    params = {
        "name": place,
        "count": 5,
        "language": "en",
        "format": "json"
    }
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "CropWise-AI/2.0"})
    with urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])
    for r in results:
        if r.get("country_code") == "PK":
            return {
                "name": r.get("name", place),
                "admin1": r.get("admin1", ""),
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
    return None

@st.cache_data(ttl=1800)
def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto"
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
        "timezone": "auto"
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

    annual = df.groupby("year", as_index=False).agg(
        Avg_Temperature_C=("temperature", "mean"),
        Rainfall_mm=("rainfall", "sum")
    )
    return annual

def weather_description(code):
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain",
        65: "Heavy rain", 80: "Rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 95: "Thunderstorm"
    }
    return codes.get(code, "Weather condition available")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🌾 CropWise AI</h1>
    <p>Predictive AI for Optimal Crop Selection — Pakistan</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "An AI-based agricultural decision-support prototype that combines "
    "crop recommendation with live and historical weather context."
)

# -----------------------------
# Location
# -----------------------------
st.sidebar.header("📍 Pakistan Location")

preset = st.sidebar.selectbox(
    "Choose a location",
    list(PAKISTAN.keys()),
    index=0
)

custom = st.sidebar.text_input(
    "Or search any Pakistan city/district",
    placeholder="e.g. Tando Allahyar"
)

if custom.strip():
    geo = geocode_pakistan(custom.strip())
    if geo:
        location_name = f'{geo["name"]}, {geo["admin1"]}'.strip(", ")
        lat, lon = geo["latitude"], geo["longitude"]
        st.sidebar.success(f"Location found: {location_name}")
    else:
        location_name = preset
        lat, lon = PAKISTAN[preset]
        st.sidebar.warning("Pakistan location not found; using selected location.")
else:
    location_name = preset
    lat, lon = PAKISTAN[preset]

st.sidebar.caption(
    "The AI model remains the same 7-feature Random Forest model. "
    "Location is used to add weather context; it does not imply field validation."
)

# -----------------------------
# Dashboard metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("AI Model", "Random Forest")
with m2:
    st.metric("Crop Classes", "22")
with m3:
    st.metric("Training Records", "2,200")
with m4:
    st.metric("Test Accuracy", "99.55%")

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
        st.metric("Temperature", f'{current["temperature_2m"]:.1f} °C')
    with w2:
        st.metric("Humidity", f'{current["relative_humidity_2m"]:.0f} %')
    with w3:
        st.metric("Rain", f'{current["rain"]:.1f} mm')
    with w4:
        st.metric("Precipitation", f'{current["precipitation"]:.1f} mm')

    st.caption(
        f'Condition: {weather_description(current["weather_code"])} | '
        f'Coordinates: {lat:.4f}, {lon:.4f} | '
        f'Weather source: Open-Meteo | Updated: {current["time"]}'
    )

    with st.expander("📅 3-Day Weather Outlook"):
        daily = weather["daily"]
        forecast = pd.DataFrame({
            "Date": daily["time"],
            "Min Temp (°C)": daily["temperature_2m_min"],
            "Max Temp (°C)": daily["temperature_2m_max"],
            "Precipitation (mm)": daily["precipitation_sum"]
        })
        st.dataframe(forecast, use_container_width=True, hide_index=True)

except Exception:
    st.warning("Live weather is temporarily unavailable. Manual inputs remain available.")

# -----------------------------
# Historical climate context
# -----------------------------
st.header("📈 Five-Year Climate Context")

try:
    hist = get_five_year_weather(lat, lon)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Temperature")
        st.line_chart(hist.set_index("year")["Avg_Temperature_C"])
    with c2:
        st.subheader("Annual Rainfall")
        st.bar_chart(hist.set_index("year")["Rainfall_mm"])
    st.caption(
        "This is historical weather/reanalysis context, not farmer field records. "
        "Open-Meteo provides historical data from global reanalysis datasets."
    )
except Exception:
    st.info("Historical weather is temporarily unavailable.")

st.divider()

# -----------------------------
# Agriculture inputs
# -----------------------------
st.header("🌱 Agricultural Conditions")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
    potassium = st.number_input("Potassium (K)", 0.0, 250.0, 50.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

st.caption(
    "Prediction inputs are kept compatible with the trained dataset: "
    "N, P, K, temperature, humidity, pH and rainfall."
)

# -----------------------------
# Soil integration
# -----------------------------
st.header("🌱 Soil Data Integration")

st.info(
    "SoilGrids is used as the planned global soil-data source for future "
    "location-based soil retrieval. The current prediction keeps soil inputs "
    "manual because the SoilGrids REST service is beta/subject to availability."
)

st.link_button("🌍 Open SoilGrids", "https://soilgrids.org/")
st.caption(
    "Source: ISRIC SoilGrids. SoilGrids provides global digital soil maps "
    "including properties such as pH and total nitrogen."
)

# -----------------------------
# Pakistan agriculture data
# -----------------------------
st.header("🇵🇰 Pakistan Agriculture Data")

st.write(
    "For future national-level analysis, CropWise AI can incorporate FAOSTAT "
    "crop production, harvested area and yield statistics. These official "
    "statistics provide national agricultural context but are not farmer-level records."
)

a1, a2 = st.columns(2)
with a1:
    st.link_button("📊 FAOSTAT Agriculture Data", "https://www.fao.org/faostat/en/")
with a2:
    st.link_button(
        "🌾 Crop Production Dataset",
        "https://data.fao.org/catalog/dataset/crop-production-yield-harvested-area-and-processed-global-national-annual-faostat"
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Best Crop", use_container_width=True, type="primary"):

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
        "Prediction Score": probabilities
    }).sort_values("Prediction Score", ascending=False).head(3)

    st.success(f"🌾 Recommended Crop: **{prediction.title()}**")

    st.header("🏆 Top 3 Crop Recommendations")

    result_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(results.reset_index(drop=True).iterrows()):
        with result_cols[i]:
            st.metric(
                f"{medals[i]} {row['Crop'].title()}",
                f"{row['Prediction Score'] * 100:.1f}%"
            )
            st.progress(float(row["Prediction Score"]))

    st.caption(
        "Prediction score = Random Forest probability estimate for the entered "
        "conditions. It is not a real-world probability of crop success."
    )

    st.header("🤖 AI Analysis")
    st.write(
        f"The model selected **{prediction.title()}** as the top recommendation "
        f"for the entered agricultural conditions."
    )
    st.write(
        "The model evaluates nitrogen, phosphorus, potassium, temperature, "
        "humidity, soil pH and rainfall together."
    )

    st.header("📊 Input Summary")
    summary = pd.DataFrame({
        "Parameter": [
            "Nitrogen", "Phosphorus", "Potassium",
            "Temperature", "Humidity", "Soil pH", "Rainfall"
        ],
        "Value": [
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

st.warning(
    "⚠️ Responsible Use: CropWise AI is a research prototype. "
    "The current Random Forest model is trained on the selected crop "
    "recommendation dataset and is not yet validated as a Pakistan-wide "
    "or Umarkot field model."
)

st.caption(
    "CropWise AI | Alibaba Cloud AI Hackathon Pakistan 2026 | "
    "Weather: Open-Meteo | Soil reference: SoilGrids | "
    "Agricultural statistics reference: FAOSTAT"
)
