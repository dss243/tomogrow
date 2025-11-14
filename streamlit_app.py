import streamlit as st
import numpy as np
from supabase import create_client

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# =====================================================
# Initialize Supabase client using st.secrets
# =====================================================
@st.cache_resource
def init_supabase():
    """
    Create a Supabase client using secrets.
    Expects SUPABASE_URL and SUPABASE_ANON_KEY in secrets.toml
    or Streamlit Cloud app secrets.
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        return None

supabase_client = init_supabase()

# =====================================================
# Simple rule-based irrigation logic
# =====================================================
def predict_irrigation_rules(temp, soil, hum, light):
    """
    Very simple rule-based decision:
    - If soil is very dry => irrigate.
    - If soil is very wet => do not irrigate.
    - Otherwise, use temperature and light to adjust.
    """
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

# =====================================================
# Fetch latest sensor data from Supabase
# =====================================================
def get_latest_data():
    """
    Get the most recent row from sensor_data table
    for device ESP32_TOMOGROW_001.
    """
    try:
        if supabase_client:
            response = (
                supabase_client
                .table("sensor_data")
                .select("*")
                .eq("device_id", "ESP32_TOMOGROW_001")
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
    return None

# =====================================================
# UI
# =====================================================
st.title("🌱 Smart Irrigation Dashboard")

latest_data = get_latest_data()

if latest_data:
    # Optional: show raw row for debugging
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

    st.subheader("🎯 Irrigation Decision (Rule-based)")
    prediction = predict_irrigation_rules(
        temperature,
        soil_moisture,
        humidity,
        light_intensity
    )

    if prediction["decision"] == "yes":
        st.error("🚨 IRRIGATION NEEDED")
    else:
        st.success("✅ NO IRRIGATION NEEDED")

    st.write(f"Confidence: {prediction['confidence']:.1%}")
else:
    st.warning("Waiting for data in Supabase (sensor_data table)...")
