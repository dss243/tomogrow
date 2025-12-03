import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# =====================================================
# Basic config
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPABASE_URL = "https://ragapkdlgtpmumwlzphs.supabase.co"
SUPABASE_ANON_KEY = "PASTE_YOUR_ANON_PUBLIC_KEY_HERE"
DEVICE_ID = "ESP32_TOMOGROW_001"

# =====================================================
# Supabase init
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
# Authentication (email / password)
# =====================================================
def ensure_auth():
    if "user" in st.session_state:
        return

    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        try:
            res = supabase_client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )  # [web:74]
            if res.user is None:
                st.error("Login failed.")
            else:
                st.session_state["user"] = res.user
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

ensure_auth()
if "user" not in st.session_state:
    st.stop()

current_user = st.session_state["user"]

# =====================================================
# Load model artifacts
# =====================================================
@st.cache_resource
def load_model_artifacts():
    model_path = "fast_tomato_irrigation_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file fast_tomato_irrigation_model.pkl was not found in the app directory.")
        return None

    try:
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

    required_keys = ["model", "scaler", "crop_encoder", "pump_encoder", "feature_names"]
    if not all(k in artifacts for k in required_keys):
        st.error("Model file does not contain all required keys.")
        return None

    return artifacts

artifacts = load_model_artifacts()

def model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    if artifacts is None:
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    crop_encoder = artifacts["crop_encoder"]
    pump_encoder = artifacts["pump_encoder"]

    # encode crop
    try:
        crop_code = crop_encoder.transform([crop_type])[0]
    except Exception as e:
        st.error(f"Error encoding crop type: {e}")
        return None

    X = np.array([[
        float(temperature),
        float(soil_moisture),
        float(humidity),
        float(light_intensity),
        crop_code,
    ]])

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

    try:
        y_enc = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

    label = pump_encoder.inverse_transform([y_enc])[0]
    conf = float(proba[y_enc])

    return {
        "irrigation_prediction": label,
        "confidence_level": round(min(conf, 0.95), 4),
        "probabilities": {
            "no": round(proba[0], 4),
            "yes": round(proba[1], 4),
        },
    }

def predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity):
    return model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato")

# =====================================================
# Data access with authorization (user_id + device_id)
# =====================================================
def get_latest_data():
    try:
        if supabase_client and current_user:
            user_id = current_user.id
            response = (
                supabase_client
                .table("sensor_data")
                .select("*")
                .eq("device_id", DEVICE_ID)
                .eq("user_id", user_id)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )  # [web:109]
            if response.data:
                return response.data[0]
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
    return None

def get_history(limit: int = 100):
    try:
        if supabase_client and current_user:
            user_id = current_user.id
            response = (
                supabase_client
                .table("sensor_data")
                .select("*")
                .eq("device_id", DEVICE_ID)
                .eq("user_id", user_id)
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            rows = response.data or []
            if not rows:
                return None
            df = pd.DataFrame(rows)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values("created_at")
            return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return None

# =====================================================
# Styling (paste your CSS here if you want)
# =====================================================
# You can paste your previous big st.markdown CSS block and header here.
# For brevity, this example keeps UI simple.

st.title("TomoGrow – Smart Irrigation Monitor")

# =====================================================
# Layout – Live + History + Simulation
# =====================================================
latest_data = get_latest_data()
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Live Field Snapshot")

    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        st.metric("Temperature (°C)", f"{temperature:.1f}")
        st.metric("Humidity (%)", f"{humidity:.1f}")
        st.metric("Soil moisture (%)", f"{soil_moisture:.1f}")
        st.metric("Light", int(light_intensity))
        st.caption(f"Last update: {timestamp}")
    else:
        st.info("No sensor data available yet for this user and device.")

    st.subheader("Irrigation Advice")
    if latest_data and artifacts is not None:
        res = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
        if res:
            decision = res["irrigation_prediction"]
            conf = res["confidence_level"]
            if decision == "yes":
                st.success("Water the plants now")
            else:
                st.info("No water needed")
            st.write(f"Confidence: {conf:.0%}")
        else:
            st.warning("Unable to generate irrigation advice.")
    else:
        st.info("Waiting for data and model to generate irrigation advice.")

with col2:
    st.subheader("Sensor History & Trends")
    points = st.slider("Data points to display", 20, 200, 80, 20)
    df_hist = get_history(limit=points)
    if df_hist is not None:
        metric_choice = st.selectbox(
            "Metric",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=2,
        )
        st.line_chart(df_hist.set_index("created_at")[metric_choice])
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No historical data yet for this user and device.")

st.subheader("Simulation Lab")

sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    sim_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
    sim_soil = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0, 1.0)
    sim_hum = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
    sim_light = st.slider("Light Intensity", 0, 1500, 500, 10)

    if artifacts is None:
        st.warning("Model not loaded; simulation unavailable.")
        sim_result = None
    else:
        sim_result = model_predict(sim_temp, sim_soil, sim_hum, sim_light, crop_type="tomato")
        if sim_result is None:
            st.error("Could not compute simulation.")
        else:
            dec = sim_result["irrigation_prediction"]
            conf = sim_result["confidence_level"]
            if dec == "yes":
                st.success(f"Simulated: Water recommended ({conf:.0%} confidence)")
            else:
                st.info(f"Simulated: No water needed ({conf:.0%} confidence)")

with sim_col2:
    st.write("Simulated Plant State")
    if artifacts is not None and sim_result is not None:
        if sim_soil > 70 and sim_result["irrigation_prediction"] == "no":
            st.success("Plant: Thriving")
        elif sim_soil < 40 or sim_result["irrigation_prediction"] == "yes":
            st.warning("Plant: Stressed – likely needs water")
        else:
            st.info("Plant: Stable growth")
    else:
        st.info("Adjust sliders and make sure model is loaded.")

