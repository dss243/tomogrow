import streamlit as st
import numpy as np
import pickle
from supabase import create_client
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# ---------------- Supabase Client ----------------
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["https://ragapkdlgtpmumwlzphs.supabase.co"]
        key = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2Fwa2RsZ3RwbXVtd2x6cGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2MTYwMDMsImV4cCI6MjA3ODE5MjAwM30.OQj-NFgd6KaDKL1BobPgLOKTCYDFmqw8KnqQFzkFWKo"]
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        return None

supabase_client = init_supabase()

# ---------------- Load ML Model ----------------
@st.cache_resource
def load_ml_model():
    model_path = "fast_tomato_irrigation_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.warning("ML model not found, using rule-based system")
        return None

ml_model_data = load_ml_model()

# ---------------- Prediction Functions ----------------
def predict_irrigation_rules(temp, soil, hum, light):
    if soil < 45:
        return {"decision": "yes", "confidence": 0.95}
    elif soil > 85:
        return {"decision": "no", "confidence": 0.95}
    elif soil < 55 and temp > 30:
        return {"decision": "yes", "confidence": 0.85}
    elif soil < 60 and light > 700:
        return {"decision": "yes", "confidence": 0.80}
    elif soil > 75 and temp < 20:
        return {"decision": "no", "confidence": 0.85}
    else:
        return {"decision": "no", "confidence": 0.75}

def predict_irrigation_ml(temp, soil, hum, light, crop="tomato"):
    if ml_model_data is None:
        return predict_irrigation_rules(temp, soil, hum, light)

    model = ml_model_data["model"]
    scaler = ml_model_data["scaler"]

    crop_mapping = {"tomato": 0, "cucumber": 1, "pepper": 2, "lettuce": 3}
    crop_encoded = crop_mapping.get(crop, 0)

    features = np.array([[temp, soil, hum, light, crop_encoded]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    conf = max(model.predict_proba(features_scaled)[0])
    decision = "yes" if pred == 1 else "no"

    return {"decision": decision, "confidence": round(conf, 2)}

# ---------------- Fetch Latest Data ----------------
def get_latest_data():
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .order("id", desc=True)\
                .limit(1)\
                .execute()
            if response.data:
                return response.data[0]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return None

# ---------------- Streamlit UI ----------------
st.title("🌱 Smart Irrigation Dashboard")

crop_type = st.sidebar.selectbox(
    "Select Crop",
    ["tomato", "cucumber", "pepper", "lettuce"]
)

latest_data = get_latest_data()

if latest_data:
    # Debug raw row (you can comment this out later)
    # st.subheader("Raw Supabase Row")
    # st.json(latest_data)

    temperature = float(latest_data.get("temperature", 0))
    humidity = float(latest_data.get("humidity", 0))
    soil_moisture = float(latest_data.get("soil_moisture", 0))
    light_intensity = float(latest_data.get("light_intensity", 0))

    st.subheader("📡 Live ESP32 Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
    col2.metric("💧 Humidity", f"{humidity:.1f} %")
    col3.metric("🌱 Soil Moisture", f"{soil_moisture:.1f} %")
    col4.metric("💡 Light Intensity", f"{int(light_intensity)}")

    st.subheader("🎯 AI Irrigation Prediction")
    prediction = predict_irrigation_ml(
        temperature,
        soil_moisture,
        humidity,
        light_intensity,
        crop_type
    )

    if prediction["decision"] == "yes":
        st.error("🚨 IRRIGATION NEEDED")
    else:
        st.success("✅ NO IRRIGATION NEEDED")

    st.write(f"Confidence: {prediction['confidence']:.1%}")
else:
    st.warning("Waiting for data in Supabase (sensor_data table)...")
