import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import os
import pandas as pd

# =====================================================
# Config - Force Light Theme with Green Colors
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force light theme with green colors
st.markdown(
    """
<style>
.stApp { background-color: #f8fdf8 !important; }
.main .block-container { background-color: #f8fdf8 !important; padding-top: 1rem; padding-bottom: 1rem; }

.stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
    color: #1a331c !important;
}

[data-testid="metric-container"] { background-color: transparent !important; }
[data-testid="metric-container"] label, [data-testid="metric-container"] div { color: #1a331c !important; }

.dataframe { background-color: white !important; color: #1a331c !important; }

.stButton button { background-color: #22c55e; color: white; }

.stAlert, .stSuccess, .stInfo, .stWarning, .stError {
    background-color: #f0f8f0 !important;
    color: #1a331c !important;
    border-left: 4px solid #22c55e;
}
</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# Config
# =====================================================
SUPABASE_URL = "https://ragapkdlgtpmumwlzphs.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2Fwa2RsZ3RwbXVtd2x6cGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2MTYwMDMsImV4cCI6MjA3ODE5MjAwM30.OQj-NFgd6KaDKL1BobPgLOKTCYDFmqw8KnqQFzkFWKo"

DEVICE_ID = "ESP32_TOMOGROW_001"

# =====================================================
# Init Supabase
# =====================================================
@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return client
    except Exception as e:
        st.error(f"Supabase connection error: {e}")
        return None

supabase_client = init_supabase()

# =====================================================
# Load model artifacts
# =====================================================
@st.cache_resource
def load_model_artifacts():
    model_path = "fast_tomato_irrigation_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file fast_tomato_irrigation_model.pkl was not found.")
        return None

    try:
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    required = ["model", "scaler", "crop_encoder", "pump_encoder", "feature_names"]
    if not all(k in artifacts for k in required):
        st.error("Missing keys in model file.")
        return None

    return artifacts

artifacts = load_model_artifacts()

# =====================================================
# Model prediction function
# =====================================================
def model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    if artifacts is None:
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    crop_encoder = artifacts["crop_encoder"]
    pump_encoder = artifacts["pump_encoder"]

    try:
        crop_code = crop_encoder.transform([crop_type])[0]
    except:
        st.error("Error encoding crop type.")
        return None

    features = np.array([
        [
            float(temperature),
            float(soil_moisture),
            float(humidity),
            float(light_intensity),
            crop_code,
        ]
    ])

    try:
        scaled = scaler.transform(features)
        encoded = model.predict(scaled)[0]
        probas = model.predict_proba(scaled)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

    decision = pump_encoder.inverse_transform([encoded])[0]
    confidence = float(probas[encoded])

    return {
        "irrigation_prediction": decision,
        "confidence_level": round(min(confidence, 0.95), 4),
        "probabilities": {
            "no": round(probas[0], 4),
            "yes": round(probas[1], 4),
        }
    }


def predict_irrigation_model_only(t, s, h, l):
    return model_predict(t, s, h, l, "tomato")

# =====================================================
# Fetch data from Supabase
# =====================================================
def get_latest_data():
    try:
        if supabase_client:
            resp = (
                supabase_client.table("sensor_data")
                .select("*")
                .eq("device_id", DEVICE_ID)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")

    return None


def get_history(limit=100):
    try:
        resp = (
            supabase_client.table("sensor_data")
            .select("*")
            .eq("device_id", DEVICE_ID)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        df = pd.DataFrame(resp.data)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values("created_at")
        return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return None

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
<div style="padding: 1.5rem; background: linear-gradient(135deg, #22c55e, #16a34a); 
border-radius: 12px; text-align:center;">
    <h1 style="color:white;">🌱 TomoGrow – Smart Irrigation Monitor</h1>
    <p style="color:#e3ffe8;">Cultivating healthier plants through intelligent irrigation</p>
</div>
""",
    unsafe_allow_html=True
)

# =====================================================
# MAIN LAYOUT
# =====================================================
latest_data = get_latest_data()

col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ----------------
with col1:
    st.markdown("### 📊 Live Field Snapshot")

    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        st.metric("🌡️ Temperature", f"{temperature}°C")
        st.metric("💧 Humidity", f"{humidity}%")
        st.metric("🌱 Soil Moisture", f"{soil_moisture}%")
        st.metric("☀️ Light", f"{light_intensity}")

        st.info(f"🕐 Last update: {timestamp}")
    else:
        st.warning("No sensor data available yet.")

    st.markdown("### 💧 Irrigation Advice")

    if latest_data and artifacts:
        result = predict_irrigation_model_only(
            temperature, soil_moisture, humidity, light_intensity
        )
        if result:
            if result["irrigation_prediction"] == "yes":
                st.success("💦 Water the plants now")
            else:
                st.info("✅ No water needed")

            st.progress(result["confidence_level"])
            st.write(f"**Confidence:** {result['confidence_level'] * 100:.1f}%")
        else:
            st.error("Prediction failed.")

# ---------------- RIGHT SIDE ----------------
with col2:
    st.markdown("### 📈 Sensor History & Trends")

    points = st.slider("Data points", 20, 200, 80, step=20)
    metric_choice = st.selectbox(
        "Metric",
        ["temperature", "humidity", "soil_moisture", "light_intensity"]
    )

    df_hist = get_history(points)

    if df_hist is not None:
        st.line_chart(df_hist.set_index("created_at")[metric_choice])
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(8),
            hide_index=True
        )
    else:
        st.warning("No history data available yet.")

# =====================================================
# SIMULATION SECTION
# =====================================================
st.markdown("---")
st.markdown("## 🔬 Simulation Lab")

sim_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
sim_soil = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0)
sim_hum = st.slider("Air Humidity (%)", 0.0, 100.0, 60.0)
sim_light = st.slider("Light Intensity", 0, 1500, 500)

if artifacts:
    sim_res = predict_irrigation_model_only(sim_temp, sim_soil, sim_hum, sim_light)

    if sim_res:
        st.subheader("Simulation Result")
        if sim_res["irrigation_prediction"] == "yes":
            st.success("💦 Water needed")
        else:
            st.info("🌿 No water needed")

        st.progress(sim_res["confidence_level"])
        st.write(f"Confidence: {sim_res['confidence_level'] * 100:.1f}%")

