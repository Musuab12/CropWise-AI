import streamlit as st
import pandas as pd
import joblib
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from datetime import date

MODEL_FILE = "cropwise_rf_model.pkl"
model = joblib.load(MODEL_FILE)

st.set_page_config(
    page_title="CropWise AI | Umarkot",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

UMARKOT_LAT = 25.3633
UMARKOT_LON = 69.7360
LOCATION_NAME = "Umarkot, Sindh, Pakistan"

# Real sample rows from the project's public crop recommendation dataset.
SCENARIOS = {
    "🌾 Rice": [90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362],
    "🌽 Maize": [71, 54, 16, 22.61359953, 63.69070564, 5.749914421, 87.75953857],
    "🫘 Chickpea": [40, 72, 77, 17.02498456, 16.98861173, 7.485996067, 88.55123143],
    "🌿 Cotton": [133, 47, 24, 24.40228894, 79.19732001, 7.231324765, 90.8022356],
    "🍌 Banana": [91, 94, 46, 29.36792366, 76.24900101, 6.149934034, 92.82840911],
    "🥭 Mango": [2, 40, 27, 29.73770045, 47.54885174, 5.954626604, 90.09586854],
    "🍎 Apple": [24, 128, 196, 22.75088787, 90.69489172, 5.521466996, 110.4317855],
}

# -----------------------------
# Premium dashboard CSS
# -----------------------------
st.markdown("""
<style>
:root {
    --green:#119447;
    --green-dark:#08783b;
    --navy:#0d1b4c;
    --muted:#60708b;
    --border:#dce5ee;
    --soft:#f5faf7;
}
.stApp { background: linear-gradient(180deg,#f7fbff 0%,#ffffff 32%,#f7faf8 100%); }
.block-container { max-width: 1500px; padding-top: .65rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#f4f8fc,#eef5f1); border-right:1px solid #dfe8e4; }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
.side-brand { font-size:1.5rem; font-weight:800; color:#073f2e; margin-bottom:.15rem; }
.side-sub { color:#617084; font-size:.82rem; margin-bottom:1rem; }
.side-location { background:#e5f6ea; border:1px solid #bfe7c9; border-radius:12px; padding:.8rem; color:#0b6f3a; font-weight:700; }
.quote { color:#516078; font-style:italic; font-size:.9rem; line-height:1.55; margin-top:2rem; padding:1rem .2rem; }

/* top banner */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; background:linear-gradient(100deg,#ffffff 0%,#f3f8fd 58%,#eaf6ef 100%); border:1px solid #dce6ed; border-radius:0 0 18px 18px; padding:.75rem 1.1rem; margin:-.65rem -0.2rem .8rem -0.2rem; box-shadow:0 5px 18px rgba(20,60,90,.05); }
.brand { display:flex; align-items:center; gap:.55rem; min-width:260px; }
.brand-icon { font-size:2rem; }
.brand-name { color:#083f34; font-size:1.65rem; font-weight:800; line-height:1; }
.brand-tag { color:#526377; font-size:.75rem; margin-top:.2rem; }
.place { color:var(--navy); font-weight:800; font-size:1.1rem; }
.place-sub { color:#596a80; font-size:.75rem; }
.heritage { text-align:right; color:#0a6e3b; font-weight:800; font-size:.9rem; }
.heritage-sub { color:#657286; font-weight:500; font-size:.7rem; }

.hero-title { font-size:2.15rem; color:var(--navy); font-weight:800; margin:.25rem 0 .05rem; }
.hero-title span { color:#087d46; }
.hero-sub { color:#4d5d73; font-size:1rem; margin-bottom:.65rem; }
.green-banner { background:linear-gradient(100deg,#139b4e,#087a3d); color:#fff; border-radius:13px; padding:.85rem 1.1rem; font-weight:700; box-shadow:0 7px 18px rgba(10,130,70,.13); }
.green-banner small { opacity:.9; font-weight:500; }

.card { background:#fff; border:1px solid var(--border); border-radius:14px; padding:1rem; box-shadow:0 4px 14px rgba(35,70,100,.045); }
.card-title { color:#132044; font-weight:800; font-size:1rem; margin-bottom:.45rem; }
.kpi { min-height:92px; }
.kpi-value { font-size:1.45rem; font-weight:800; color:#111a2e; }
.kpi-label { color:#637188; font-size:.75rem; }
.kpi-green { color:#0a9a4c; }
.kpi-blue { color:#1673ea; }
.kpi-red { color:#df2020; }

.section-title { color:#101b3c; font-weight:800; font-size:1.12rem; margin:.25rem 0 .35rem; }
.small-note { color:#65748b; font-size:.78rem; }

/* inputs */
div[data-testid="stNumberInput"] label { color:#17213e; font-weight:700; font-size:.78rem; }
div[data-testid="stNumberInput"] input { border-radius:9px; }

/* buttons */
.stButton > button { border-radius:9px; font-weight:700; border:1px solid #dce4eb; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#0b9a4d,#078341); color:#fff; border:none; }
.demo-caption { color:#66758a; font-size:.78rem; margin-bottom:.45rem; }

/* results */
.reco-wrap { background:linear-gradient(135deg,#ecfff1,#f6fff8); border:1px solid #ccebd4; border-radius:15px; padding:1rem; }
.reco-card { background:#fff; border:1px solid #d8e7de; border-radius:13px; padding:.9rem; text-align:center; min-height:190px; }
.medal { font-size:1.45rem; }
.crop-name { color:#0f1d42; font-size:1.08rem; font-weight:800; margin:.25rem 0; }
.score { color:#087d46; font-size:1.15rem; font-weight:800; }
.why-card { background:#fff; border:1px solid var(--border); border-radius:14px; padding:1rem; }
.why-line { margin:.45rem 0; color:#3f4f68; font-size:.82rem; }
.check { color:#0a9d4c; font-weight:900; }
.responsible { background:#e9f5ff; border:1px solid #cde5f9; border-radius:14px; padding:1rem; color:#203a5e; font-size:.82rem; line-height:1.55; }
.future { background:#eaf8ef; border:1px solid #cbe9d4; border-radius:14px; padding:1rem; color:#355344; font-size:.8rem; line-height:1.5; }
.footer { text-align:center; color:#66758b; font-size:.75rem; padding:1rem 0 .25rem; }

/* hide streamlit chrome where useful */
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data functions
# -----------------------------
@st.cache_data(ttl=1800)
def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,apparent_temperature,wind_speed_10m,surface_pressure",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "CropWise-AI/3.0"})
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
    req = Request(url, headers={"User-Agent": "CropWise-AI/3.0"})
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
    codes = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Rime fog",51:"Light drizzle",53:"Moderate drizzle",55:"Dense drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",80:"Rain showers",81:"Moderate rain showers",82:"Violent rain showers",95:"Thunderstorm"}
    return codes.get(code, "Weather condition available")

def apply_scenario(label):
    vals = SCENARIOS[label]
    keys = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for k, v in zip(keys, vals):
        st.session_state[k] = float(v)
    st.session_state["selected_demo"] = label

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="side-brand">🌱 CropWise AI</div><div class="side-sub">AI for Smart Agriculture</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-location">📍 Umarkot, Sindh, Pakistan<br><span style="font-size:.72rem;font-weight:500">Local research focus</span></div>', unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.markdown("🏠 **Home**")
st.sidebar.markdown("☁️ Weather & Climate")
st.sidebar.markdown("🌱 Soil & Inputs")
st.sidebar.markdown("📊 Crop Prediction")
st.sidebar.markdown("🗃️ Data & Sources")
st.sidebar.markdown("ℹ️ About Project")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="quote">“Empowering farmers with AI for a food-secure Pakistan.”</div>', unsafe_allow_html=True)

# -----------------------------
# Top banner
# -----------------------------
st.markdown('''
<div class="topbar">
  <div class="brand"><div class="brand-icon">🌿</div><div><div class="brand-name">CropWise AI</div><div class="brand-tag">AI for Smart Agriculture</div></div></div>
  <div><div class="place">📍 Umarkot, Sindh, Pakistan</div><div class="place-sub">Local Insights for a Sustainable Tomorrow</div></div>
  <div class="heritage">Umarkot<br><span class="heritage-sub">Heritage | Agriculture | Hope</span></div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="hero-title">🌱 Welcome to <span>CropWise AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered crop recommendation for Umarkot, Sindh, Pakistan<br>Enter soil parameters and get the most suitable crops using machine learning and real weather context.</div>', unsafe_allow_html=True)

banner = st.columns([3,1.7])
with banner[0]:
    st.markdown('<div class="green-banner">🌱 Sustainable Agriculture for a Brighter Tomorrow<br><small>“From Data to Better Harvests”</small></div>', unsafe_allow_html=True)
with banner[1]:
    st.markdown('<div class="card" style="text-align:center"><b>Research Prototype</b><br><span class="small-note">Umarkot-first • AI decision support</span></div>', unsafe_allow_html=True)

# -----------------------------
# Live KPI cards
# -----------------------------
try:
    weather = get_weather(UMARKOT_LAT, UMARKOT_LON)
    current = weather["current"]
    temp = current.get("temperature_2m")
    humidity_live = current.get("relative_humidity_2m")
    rain_live = current.get("rain", current.get("precipitation", 0))
    condition = weather_description(current.get("weather_code", 0))
except Exception:
    weather = None
    temp, humidity_live, rain_live, condition = None, None, None, "Live data unavailable"

kpi = st.columns(5)
items = [
    ("🌡️", f"{temp:.1f}°C" if temp is not None else "—", "Current Temperature", "Umarkot (Live)", "kpi-red"),
    ("💧", f"{humidity_live:.0f}%" if humidity_live is not None else "—", "Humidity", "Umarkot (Live)", "kpi-blue"),
    ("🌧️", f"{rain_live:.1f} mm" if rain_live is not None else "—", "Rainfall (Today)", "Umarkot (Live)", "kpi-blue"),
    ("🌿", "Arid / Semi-Arid", "Climate Type", "Umarkot", "kpi-green"),
    ("📅", "5 Years", "Historical Context", "Weather / Reanalysis", "kpi-green"),
]
for col, (icon, value, label, sub, cls) in zip(kpi, items):
    with col:
        st.markdown(f'<div class="card kpi"><div style="font-size:1.25rem">{icon}</div><div class="kpi-value {cls}">{value}</div><div class="kpi-label">{label}<br>{sub}</div></div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Main data panels
# -----------------------------
left, mid, right = st.columns([1.15,1.65,1.35])

with left:
    st.markdown('<div class="card" style="min-height:285px"><div class="card-title">☁️ Live Weather - Umarkot</div>', unsafe_allow_html=True)
    if weather:
        st.markdown(f'<div style="font-size:3rem;text-align:center">☀️</div><div style="text-align:center;font-size:1.75rem;font-weight:800;color:#18213d">{temp:.1f}°C</div><div style="text-align:center;color:#44536b;margin:.25rem 0 .7rem">{condition}</div>', unsafe_allow_html=True)
        st.write(f"Feels like: {current.get('apparent_temperature', temp):.1f}°C")
        st.write(f"Humidity: {humidity_live:.0f}%")
        st.write(f"Wind Speed: {current.get('wind_speed_10m', 0):.1f} km/h")
        st.write(f"Pressure: {current.get('surface_pressure', 0):.0f} hPa")
        st.caption("Source: Open-Meteo (Live Data)")
    else:
        st.warning("Live weather is temporarily unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

with mid:
    st.markdown('<div class="card" style="min-height:285px"><div class="card-title">📊 5-Year Climate Trend (Umarkot)</div>', unsafe_allow_html=True)
    try:
        hist = get_five_year_weather(UMARKOT_LAT, UMARKOT_LON)
        chart = hist.set_index("year")[["Avg_Temperature_C", "Rainfall_mm"]].rename(columns={"Avg_Temperature_C":"Avg Temperature (°C)","Rainfall_mm":"Total Rainfall (mm)"})
        st.line_chart(chart, height=185)
        st.caption("Source: Open-Meteo Historical / Reanalysis Data")
    except Exception:
        st.info("Historical climate data is temporarily unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card" style="min-height:285px"><div class="card-title">🌱 Soil Input Parameters</div>', unsafe_allow_html=True)
    defaults = {"N":90.0,"P":42.0,"K":43.0,"temperature":25.0,"humidity":70.0,"ph":6.5,"rainfall":120.0}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
    r1 = st.columns(3)
    with r1[0]: nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, key="N", step=1.0)
    with r1[1]: phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, key="P", step=1.0)
    with r1[2]: potassium = st.number_input("Potassium (K)", 0.0, 250.0, key="K", step=1.0)
    r2 = st.columns(2)
    with r2[0]: ph = st.number_input("Soil pH", 0.0, 14.0, key="ph", step=.1)
    with r2[1]: rainfall = st.number_input("Expected Rainfall", 0.0, 500.0, key="rainfall", step=1.0)
    r3 = st.columns(2)
    with r3[0]: temperature = st.number_input("Temperature (°C)", 0.0, 50.0, key="temperature", step=.1)
    with r3[1]: humidity = st.number_input("Humidity (%)", 0.0, 100.0, key="humidity", step=1.0)
    predict = st.button("⚙️ Predict Best Crops", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Demo scenarios
# -----------------------------
st.write("")
st.markdown('<div class="card"><div class="section-title">🧪 Quick Demo Scenarios <span class="small-note">(Load example values)</span></div><div class="demo-caption">These are real sample conditions from the public training dataset — demonstration only, not farmer records.</div>', unsafe_allow_html=True)
cols = st.columns(7)
for i, label in enumerate(SCENARIOS.keys()):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"demo_{i}"):
            apply_scenario(label)
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction result
# -----------------------------
if predict:
    input_data = pd.DataFrame([{"N":nitrogen,"P":phosphorus,"K":potassium,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    results = pd.DataFrame({"Crop":model.classes_,"Prediction Score":probabilities}).sort_values("Prediction Score", ascending=False).head(3).reset_index(drop=True)

    st.write("")
    res_left, res_right = st.columns([2.05,1])
    with res_left:
        st.markdown('<div class="reco-wrap"><div class="section-title">🏆 Top Crop Recommendations</div><div class="small-note">Based on the entered soil, weather and climate conditions for Umarkot</div>', unsafe_allow_html=True)
        cards = st.columns(3)
        medals=["🥇","🥈","🥉"]
        for i,row in results.iterrows():
            with cards[i]:
                st.markdown(f'<div class="reco-card"><div class="medal">{medals[i]}</div><div class="crop-name">{row["Crop"].title()}</div><div class="score">Confidence: {row["Prediction Score"]*100:.1f}%</div><div class="small-note" style="margin-top:.5rem">Model prediction score</div></div>', unsafe_allow_html=True)
                st.progress(float(row["Prediction Score"]))
        st.markdown('</div>', unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="why-card"><div class="section-title">💡 Why these crops?</div><div class="why-line"><span class="check">✓</span> Match with soil nutrient levels</div><div class="why-line"><span class="check">✓</span> Compare temperature and humidity</div><div class="why-line"><span class="check">✓</span> Consider rainfall conditions</div><div class="why-line"><span class="check">✓</span> Higher model score among 22 classes</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="responsible" style="margin-top:.7rem"><b>🔵 Responsible Use</b><br>CropWise AI is a research prototype. The current model is trained on a public crop recommendation dataset and is not yet field-validated with Umarkot farmer data.</div>', unsafe_allow_html=True)

    st.info(f"🌾 Model recommendation: **{prediction.title()}**. The current Random Forest uses 7 features: N, P, K, temperature, humidity, pH and rainfall.")

# -----------------------------
# Data and future section
# -----------------------------
st.write("")
st.markdown('<div class="section-title">🗃️ Data & Future Integration</div>', unsafe_allow_html=True)
d1,d2,d3 = st.columns(3)
with d1:
    st.markdown('<div class="card"><b>☁️ Weather</b><br><span class="small-note">Live and historical data from Open-Meteo.</span></div>', unsafe_allow_html=True)
with d2:
    st.markdown('<div class="card"><b>🌱 Soil</b><br><span class="small-note">Global soil-data reference (SoilGrids) for future validated integration.</span></div>', unsafe_allow_html=True)
with d3:
    st.markdown('<div class="card"><b>🌾 Agriculture</b><br><span class="small-note">Umarkot and Sindh agricultural data for future localized analysis.</span></div>', unsafe_allow_html=True)

st.markdown('<div class="future" style="margin-top:.7rem"><b>📈 Future Development</b><br>Validate CropWise AI with Umarkot-specific field soil data, farmer records, localized crop-yield data and additional AI models before expanding to other regions of Pakistan.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">CropWise AI • AI for Smart Agriculture • Umarkot, Sindh, Pakistan • Alibaba Cloud AI Hackathon Pakistan 2026 &nbsp; | &nbsp; 🌱 A Greener Pakistan | A Better Tomorrow</div>', unsafe_allow_html=True)
