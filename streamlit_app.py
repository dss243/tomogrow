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
# Load full model artifacts (model, scaler, encoders)
# =====================================================
@st.cache_resource
def load_model_artifacts():
    model_path = "fast_tomato_irrigation_model.pkl"
    if not os.path.exists(model_path):
        st.error("ML model file fast_tomato_irrigation_model.pkl not found in app directory.")
        return None

    try:
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None

    required_keys = ["model", "scaler", "crop_encoder", "pump_encoder", "feature_names"]
    if not all(k in artifacts for k in required_keys):
        st.error(f"Model pickle is missing some required keys: {required_keys}")
        return None

    return artifacts

artifacts = load_model_artifacts()

# =====================================================
# Prediction function – same logic as training script
# =====================================================
def predict_irrigation(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    """
    Reimplements predict_irrigation_regularized using loaded artifacts.
    """
    if artifacts is None:
        st.error("Model artifacts not loaded; cannot predict.")
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    crop_encoder = artifacts["crop_encoder"]
    pump_encoder = artifacts["pump_encoder"]

    # Build input_data dict like in your training code
    input_data = {
        "Crop_Type": crop_type,
        "Temperature": float(temperature),
        "Soil_Moisture": float(soil_moisture),
        "Humidity": float(humidity),
        "Light_Intensity": float(light_intensity),
    }

    # Build features array with 5 features
    features = np.array([[
        input_data["Temperature"],
        input_data["Soil_Moisture"],
        input_data["Humidity"],
        input_data["Light_Intensity"],
        crop_encoder.transform([input_data["Crop_Type"]])[0]
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
        st.error(f"Model prediction error: {e}")
        return None

    prediction = pump_encoder.inverse_transform([prediction_encoded])[0]
    confidence = float(probabilities[prediction_encoded])

    # Decision logic (same as your code)
    soil_moisture = input_data["Soil_Moisture"]
    temperature = input_data["Temperature"]

    irrigation_decision = prediction
    if soil_moisture < 45:
        irrigation_decision = "yes"
    elif soil_moisture > 85:
        irrigation_decision = "no"
    elif soil_moisture < 55 and temperature > 30:
        irrigation_decision = "yes"
    elif soil_moisture > 75 and temperature < 20:
        irrigation_decision = "no"

    final_confidence = min(confidence, 0.95)

    return {
        "irrigation_prediction": prediction,
        "irrigation_decision": irrigation_decision,
        "confidence_level": round(final_confidence, 4),
        "soil_moisture_level": soil_moisture,
        "model_used": type(model).__name__,
        "probabilities": {
            "no": round(probabilities[0], 4),
            "yes": round(probabilities[1], 4),
        },
    }

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
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
                df = df.sort_values("created_at")
            return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return None

# =====================================================
# UI styles
# =====================================================
st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #888;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        padding: 0.8rem 0.6rem 0.2rem 0.6rem;
        border-radius: 0.6rem;
        background-color: #f7f9fb;
        border: 1px solid #e3e7ed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🍅 TomoGrow – Smart Irrigation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ESP32 ➜ ThingSpeak ➜ Supabase ➜ ML model ➜ Irrigation decision</div>', unsafe_allow_html=True)

tabs = st.tabs(["🌟 Live", "📈 History", "ℹ️ About"])

# =====================================================
# LIVE TAB
# =====================================================
with tabs[0]:
    latest_data = get_latest_data()

    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        # Crop fixed to "tomato" since this is TomoGrow
        crop_type = "tomato"

        top_col1, top_col2 = st.columns([2, 1])

        with top_col1:
            st.markdown("#### 📡 Live ESP32 Measurements")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("🌡️ Temperature", f"{temperature:.1f} °C")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("💧 Humidity", f"{humidity:.1f} %")
                st.markdown("</div>", unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("🌱 Soil Moisture", f"{soil_moisture:.1f} %")
                st.markdown("</div>", unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("💡 Light", f"{int(light_intensity)}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"Last update: {timestamp}")

        with top_col2:
            st.markdown("#### 🎯 ML Irrigation Decision")

            result = predict_irrigation(
                temperature,
                soil_moisture,
                humidity,
                light_intensity,
                crop_type=crop_type,
            )

            if result is None:
                st.warning("Prediction is not available.")
            else:
                decision = result["irrigation_decision"]
                conf = result["confidence_level"]

                if decision == "yes":
                    st.error("🚨 Irrigation needed")
                else:
                    st.success("✅ No irrigation needed")

                st.write(f"Model prediction: `{result['irrigation_prediction']}`")
                st.write(f"Confidence: **{conf:.1%}**")
                st.caption(f"Model used: {result['model_used']}")

    else:
        st.warning("Waiting for data in Supabase (sensor_data table)...")

# =====================================================
# HISTORY TAB
# =====================================================
with tabs[1]:
    st.subheader("📈 Sensor History")

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
            height=320
        )

        st.markdown("#### Recent data table")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]],
            use_container_width=True,
            hide_index=True
        )

# =====================================================
# ABOUT TAB
# =====================================================
with tabs[2]:
    st.subheader("ℹ️ About TomoGrow")
    st.markdown(
        """
        **TomoGrow** is a smart irrigation demo that connects:

        - An ESP32 with sensors (temperature, humidity, soil moisture, light)  
        - ThingSpeak as a simple IoT buffer  
        - Supabase as a cloud database  
        - A RandomForest/Logistic Regression ML pipeline saved as `fast_tomato_irrigation_model.pkl`  

        The web app uses the same scaler, encoders, and decision logic as in the training notebook to keep predictions consistent.
        """
    )
