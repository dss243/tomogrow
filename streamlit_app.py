import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import os
import pandas as pd

# =====================================================
# Config - Force Light Theme
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force light theme
st.markdown(
    """
    <style>
    /* Force complete light theme */
    .stApp {
        background-color: white !important;
    }
    
    .main .block-container {
        background-color: white !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Make all text dark */
    .stMarkdown, .stText, .stWrite, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    
    /* Fix metric colors */
    [data-testid="metric-container"] {
        background-color: transparent !important;
    }
    
    [data-testid="metric-container"] label, [data-testid="metric-container"] div {
        color: #333333 !important;
    }
    
    /* Fix dataframe colors */
    .dataframe {
        background-color: white !important;
        color: #333333 !important;
    }
    
    /* Fix slider and selectbox colors */
    .stSlider, .stSelectbox {
        background-color: white !important;
        color: #333333 !important;
    }
    
    /* Fix card backgrounds */
    .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
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

# =====================================================
# Prediction – pure model decision (reusable)
# =====================================================
def model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    if artifacts is None:
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    crop_encoder = artifacts["crop_encoder"]
    pump_encoder = artifacts["pump_encoder"]

    input_data = {
        "Crop_Type": crop_type,
        "Temperature": float(temperature),
        "Soil_Moisture": float(soil_moisture),
        "Humidity": float(humidity),
        "Light_Intensity": float(light_intensity),
    }

    try:
        crop_code = crop_encoder.transform([input_data["Crop_Type"]])[0]
    except Exception as e:
        st.error(f"Error encoding crop type: {e}")
        return None

    features = np.array([[
        input_data["Temperature"],
        input_data["Soil_Moisture"],
        input_data["Humidity"],
        input_data["Light_Intensity"],
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
        "irrigation_prediction": prediction_label,      # yes / no from model
        "confidence_level": round(min(confidence, 0.95), 4),
        "probabilities": {
            "no": round(probabilities[0], 4),
            "yes": round(probabilities[1], 4),
        },
    }

# Wrapper for live data
def predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity):
    return model_predict(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato")

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


def get_history(limit: int = 100):
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
# Clean Light Styling
# =====================================================
st.markdown(
    """
    <style>
    /* Clean light theme */
    .header {
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c5530;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: #666666;
        margin-top: 0.25rem;
    }
    
    /* Card styling */
    .card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 0.75rem;
    }
    
    /* Metric grid */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c5530;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666666;
        margin-top: 0.25rem;
    }
    
    /* Status text */
    .status-text {
        font-size: 0.85rem;
        color: #666666;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <div class="header-title">TomoGrow – Smart Irrigation Monitor</div>
        <div class="header-subtitle">Real-time monitoring for optimal plant care</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Main Layout
# =====================================================
latest_data = get_latest_data()

col1, col2 = st.columns([1, 1])

# ---------------------- LEFT COLUMN ----------------------
with col1:
    # Live Field Snapshot
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Live Field Snapshot</div>', unsafe_allow_html=True)
    
    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")
        
        # Metric grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        # Temperature
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Temperature</div>
            <div class="metric-value">{}°C</div>
        </div>
        """.format(temperature), unsafe_allow_html=True)
        
        # Humidity
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Humidity</div>
            <div class="metric-value">{}%</div>
        </div>
        """.format(humidity), unsafe_allow_html=True)
        
        # Soil Moisture
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Soil Moisture</div>
            <div class="metric-value">{}%</div>
        </div>
        """.format(soil_moisture), unsafe_allow_html=True)
        
        # Light
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Light</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(int(light_intensity)), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Timestamp
        st.markdown(
            f'<div class="status-text">Last update from the field: {timestamp}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No sensor data available. Data will appear when the device starts sending.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Irrigation Advice
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Irrigation Advice</div>', unsafe_allow_html=True)
    
    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
        
        if result:
            decision = result["irrigation_prediction"]
            conf = result["confidence_level"]
            
            if decision == "yes":
                st.success("**Water the plants now**")
                st.write("Current conditions suggest watering would benefit the plants.")
            else:
                st.info("**No water needed**")
                st.write("Conditions are comfortable for the plants.")
            
            st.write(f"Confidence: {conf:.0%}")
        else:
            st.write("Unable to generate irrigation advice.")
    else:
        st.write("Waiting for data and model to generate advice.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- RIGHT COLUMN ----------------------
with col2:
    # Sensor History & Trends
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Sensor History & Trends</div>', unsafe_allow_html=True)
    
    points = st.slider(
        "Number of data points to display",
        min_value=20,
        max_value=200,
        value=80,
        step=20,
    )
    
    df_hist = get_history(limit=points)
    
    if df_hist is not None:
        metric_choice = st.selectbox(
            "Select metric to visualize",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=2,
        )
        
        # Display chart
        st.line_chart(
            df_hist.set_index("created_at")[metric_choice],
            height=300
        )
        
        # Recent data table
        st.markdown("**Recent Measurements**")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No historical data available yet.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- BOTTOM SECTION ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Plant Health Status</div>', unsafe_allow_html=True)

if latest_data and artifacts is not None:
    result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
    
    if result:
        decision = result["irrigation_prediction"]
        
        if soil_moisture > 70 and decision == "no":
            status = "🌿 Healthy"
            message = "Plant is in good condition with adequate moisture."
        elif soil_moisture < 40 or decision == "yes":
            status = "💧 Needs Attention"
            message = "Plant may need watering soon."
        else:
            status = "🌱 Stable"
            message = "Plant is doing well under current conditions."
        
        col_status, col_message = st.columns([1, 3])
        with col_status:
            st.metric("Status", status)
        with col_message:
            st.write(message)
    else:
        st.write("Unable to determine plant health status.")
else:
    st.write("Waiting for data to assess plant health.")

st.markdown("</div>", unsafe_allow_html=True)
