import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# =====================================================
# Config
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
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
# Authentication
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
        st.error("Model file does not contain all required keys: model, scaler, crop_encoder, pump_encoder, feature_names.")
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

    try:
        crop_code = crop_encoder.transform([crop_type])[0]
    except Exception as e:
        st.error(f"Error encoding crop type: {e}")
        return None

    features = np.array([[
        float(temperature),
        float(soil_moisture),
        float(humidity),
        float(light_intensity),
        crop_code,
    ]])

    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

    try:
        prediction_encoded = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

    prediction_label = pump_encoder.inverse_transform([prediction_encoded])[0]
    confidence = float(probabilities[prediction_encoded])

    return {
        "irrigation_prediction": prediction_label,
        "confidence_level": round(min(confidence, 0.95), 4),
        "probabilities": {
            "no": round(probabilities[0], 4),
            "yes": round(probabilities[1], 4),
        },
    }

def predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity):
    return model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato")

# =====================================================
# Fetch data from Supabase (per-user)
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
            )  # [web:69][web:78]
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
            data = response.data or []
            if not data:
                return None
            df = pd.DataFrame(data)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values("created_at")
            return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return None

# =====================================================
# Styling (same as you already have)
# =====================================================
# keep your existing CSS & layout here (omitted to save space)
# paste your big st.markdown CSS block and UI sections exactly as before

# For brevity, only showing the data usage part:

latest_data = get_latest_data()
col1, col2 = st.columns([1, 1])

with col1:
    # Live Field Snapshot
    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        # ... your nice HTML cards here ...
    else:
        st.info("No sensor data available for this user and device.")

    # Irrigation Advice
    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
        if result:
            decision = result["irrigation_prediction"]
            conf = result["confidence_level"]
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
    st.subheader("History")
    points = st.slider("Data points to display", 20, 200, 80, 20)
    df_hist = get_history(limit=points)
    if df_hist is not None:
        st.line_chart(df_hist.set_index("created_at")["soil_moisture"])
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No historical data for this user yet.")
