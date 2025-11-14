import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import os
import pandas as pd

# =====================================================
# Config
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    page_icon="🍅",
    layout="wide"
)

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
        st.error(f"Supabase init failed: {e}")
        return None

supabase_client = init_supabase()

# =====================================================
# Load ML model (model only, ignore scaler)
# =====================================================
@st.cache_resource
def load_ml_model():
    model_path = "fast_tomato_irrigation_model.pkl"
    if not os.path.exists(model_path):
        st.error("ML model file fast_tomato_irrigation_model.pkl not found in app directory.")
        return None

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None

    # If you saved a dict {"model": clf, "scaler": scaler}, take only the model
    if isinstance(model_data, dict):
        model = model_data.get("model", None)
    else:
        model = model_data

    if model is None:
        st.error("ML model object not found inside the pickle file.")
        return None

    return model

ml_model = load_ml_model()

# =====================================================
# ML prediction (no scaler, no rules)
# =====================================================
def predict_irrigation_ml(temp, soil, hum, light):
    if ml_model is None:
        st.error("ML model is not loaded; cannot make predictions.")
        return {"decision": "no", "confidence": 0.0}

    # Adjust features to match your training (here: [temp, soil, hum, light])
    X = np.array([[temp, soil, hum, light]])

    try:
        pred = ml_model.predict(X)[0]
        if hasattr(ml_model, "predict_proba"):
            conf = float(np.max(ml_model.predict_proba(X)))
        else:
            conf = 0.8
        decision = "yes" if int(pred) == 1 else "no"
        return {"decision": decision, "confidence": round(conf, 2)}
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        return {"decision": "no", "confidence": 0.0}

# =====================================================
# Fetch data from Supabase
# =====================================================
def get_latest_data():
    try:
        if supabase_client:
            response = (
                supabase_client
                .table("sensor_data")
                .select("*")
                .eq("device_id", DEVICE_ID)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
    return None


def get_history(limit: int = 50):
    """Fetch last N records for history view."""
    try:
        if supabase_client:
            response = (
                supabase_client
                .table("sensor_data")
                .select("*")
                .eq("device_id", DEVICE_ID)
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            data = response.data or []
            if not data:
                return None
            df = pd.DataFrame(data)
            # Ensure created_at is datetime and sort oldest -> newest
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
                df = df.sort_values("created_at")
            return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return None

# =====================================================
# UI Layout – tabs
# =====================================================
st.title("🍅 TomoGrow – Smart Irrigation Dashboard")

tabs = st.tabs(["🌟 Live", "📈 History", "ℹ️ About"])

# ---------------------- LIVE TAB ----------------------
with tabs[0]:
    st.subheader("Real‑Time Status")

    latest_data = get_latest_data()

    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        top_col1, top_col2 = st.columns([2, 1])

        with top_col1:
            st.markdown("#### 📡 Live ESP32 Measurements")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
            col2.metric("💧 Humidity", f"{humidity:.1f} %")
            col3.metric("🌱 Soil Moisture", f"{soil_moisture:.1f} %")
            col4.metric("💡 Light", f"{int(light_intensity)}")
            st.caption(f"Last update: {timestamp}")

        with top_col2:
            st.markdown("#### 🎯 ML Irrigation Decision")
            prediction = predict_irrigation_ml(
                temperature,
                soil_moisture,
                humidity,
                light_intensity
            )

            if prediction["decision"] == "yes":
                st.error("🚨 Irrigation needed")
            else:
                st.success("✅ No irrigation needed")

            st.write(f"Model confidence: **{prediction['confidence']:.1%}**")

    else:
        st.warning("Waiting for data in Supabase (sensor_data table)...")

# ---------------------- HISTORY TAB ----------------------
with tabs[1]:
    st.subheader("Sensor History")

    col_hist1, col_hist2 = st.columns([3, 1])
    with col_hist2:
        points = st.slider("Number of points", min_value=10, max_value=200, value=50, step=10)

    df_hist = get_history(limit=points)

    if df_hist is None:
        st.info("No history yet. Keep the ESP32 and bridge running to collect data.")
    else:
        metric_choice = st.selectbox(
            "Select metric to visualize",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=0
        )

        st.line_chart(
            df_hist.set_index("created_at")[metric_choice],
            height=300
        )

        st.markdown("#### Recent data table")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]],
            use_container_width=True,
            hide_index=True
        )

# ---------------------- ABOUT TAB ----------------------
with tabs[2]:
    st.subheader("About TomoGrow")
    st.markdown(
        """
        **TomoGrow** is a smart irrigation demo that connects:

        - An ESP32 with sensors (temperature, humidity, soil moisture, light)  
        - ThingSpeak as a simple IoT buffer  
        - Supabase as a cloud database  
        - A machine‑learning model (`fast_tomato_irrigation_model.pkl`) to decide when to irrigate  

        This dashboard shows the latest sensor readings, the model's decision, and a short history of recent measurements.
        """
    )
