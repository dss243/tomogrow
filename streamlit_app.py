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
    /* Force complete light theme with green accents */
    .stApp {
        background-color: #f8fdf8 !important;
    }
    
    .main .block-container {
        background-color: #f8fdf8 !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Make all text dark */
    .stMarkdown, .stText, .stWrite, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #1a331c !important;
    }
    
    /* Fix metric colors with green theme */
    [data-testid="metric-container"] {
        background-color: transparent !important;
    }
    
    [data-testid="metric-container"] label, [data-testid="metric-container"] div {
        color: #1a331c !important;
    }
    
    /* Fix dataframe colors */
    .dataframe {
        background-color: white !important;
        color: #1a331c !important;
    }
    
    /* Fix slider and selectbox colors */
    .stSlider, .stSelectbox {
        background-color: white !important;
        color: #1a331c !important;
    }
    
    /* Green alerts */
    .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #f0f8f0 !important;
        color: #1a331c !important;
        border-left: 4px solid #22c55e;
    }
    
    /* Green progress bars */
    .stProgress > div > div {
        background-color: #22c55e;
    }
    
    /* Green buttons */
    .stButton button {
        background-color: #22c55e;
        color: white;
    }
    
    /* Green slider */
    .stSlider > div > div {
        background-color: #22c55e;
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
# Beautiful Green Styling
# =====================================================
st.markdown(
    """
    <style>
    /* Green theme */
    .header {
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card styling with green accents */
    .card {
        background: white;
        border: 1px solid #dcfce7;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.08);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.12);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #166534;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-title::before {
        content: "🌿";
        font-size: 1.3em;
    }
    
    /* Metric grid with green theme */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #bbf7d0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #22c55e;
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #166534;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #4d7c0f;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status text */
    .status-text {
        font-size: 0.85rem;
        color: #4d7c0f;
        font-style: italic;
        background: #f0fdf4;
        padding: 0.5rem;
        border-radius: 6px;
        border-left: 3px solid #22c55e;
    }
    
    /* Plant status indicators */
    .plant-status-healthy {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
        color: #166534;
    }
    
    .plant-status-attention {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        color: #92400e;
    }
    
    .plant-status-stable {
        background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
        border: 2px solid #3b82f6;
        color: #1e40af;
    }
    
    /* Section dividers */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #22c55e 50%, transparent 100%);
        margin: 2rem 0;
        opacity: 0.3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <div class="header-title">🌱 TomoGrow – Smart Irrigation Monitor</div>
        <div class="header-subtitle">Cultivating healthier plants through intelligent irrigation</div>
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
    st.markdown('<div class="card-title">📊 Live Field Snapshot</div>', unsafe_allow_html=True)
    
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
            <div class="metric-label">🌡️ Temperature</div>
            <div class="metric-value">{}°C</div>
        </div>
        """.format(temperature), unsafe_allow_html=True)
        
        # Humidity
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">💧 Humidity</div>
            <div class="metric-value">{}%</div>
        </div>
        """.format(humidity), unsafe_allow_html=True)
        
        # Soil Moisture
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🌱 Soil Moisture</div>
            <div class="metric-value">{}%</div>
        </div>
        """.format(soil_moisture), unsafe_allow_html=True)
        
        # Light
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">☀️ Light</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(int(light_intensity)), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Timestamp
        st.markdown(
            f'<div class="status-text">🕐 Last update from the field: {timestamp}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("📡 No sensor data available. Data will appear when the device starts sending.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Irrigation Advice
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💧 Irrigation Advice</div>', unsafe_allow_html=True)
    
    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
        
        if result:
            decision = result["irrigation_prediction"]
            conf = result["confidence_level"]
            
            if decision == "yes":
                st.success("**💦 Water the plants now**")
                st.write("Current conditions suggest watering would benefit the plants for optimal growth.")
                st.progress(conf)
            else:
                st.info("**✅ No water needed**")
                st.write("Conditions are comfortable for the plants. Continue monitoring.")
                st.progress(conf)
            
            st.write(f"**Confidence Level:** {conf:.0%}")
        else:
            st.warning("⚠️ Unable to generate irrigation advice at this time.")
    else:
        st.info("⏳ Waiting for data and model to generate irrigation advice.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- RIGHT COLUMN ----------------------
with col2:
    # Sensor History & Trends
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Sensor History & Trends</div>', unsafe_allow_html=True)
    
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
            format_func=lambda x: {
                "temperature": "🌡️ Temperature",
                "humidity": "💧 Humidity", 
                "soil_moisture": "🌱 Soil Moisture",
                "light_intensity": "☀️ Light Intensity"
            }[x]
        )
        
        # Display chart
        st.line_chart(
            df_hist.set_index("created_at")[metric_choice],
            height=320
        )
        
        # Recent data table
        st.markdown("**Recent Measurements**")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("📊 No historical data available yet. Data will accumulate over time.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------------- BOTTOM SECTION ----------------------
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌿 Plant Health Status</div>', unsafe_allow_html=True)

    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
        
        if result:
            decision = result["irrigation_prediction"]
            
            if soil_moisture > 70 and decision == "no":
                status_class = "plant-status-healthy"
                status = "🌿 Vibrant & Healthy"
                message = "Your plants are thriving with optimal moisture levels and showing vigorous growth."
            elif soil_moisture < 40 or decision == "yes":
                status_class = "plant-status-attention"
                status = "💧 Needs Attention"
                message = "Plants show signs of stress. Consider watering to maintain optimal health."
            else:
                status_class = "plant-status-stable"
                status = "🌱 Stable & Growing"
                message = "Plants are maintaining steady growth under current environmental conditions."
            
            st.markdown(f"""
            <div class="metric-card {status_class}" style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{status.split(' ')[0]}</div>
                <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; color: inherit;">{status}</div>
                <div style="font-size: 0.9rem; color: inherit;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔍 Analyzing plant health data...")
    else:
        st.info("🌱 Plant health assessment will appear when sensor data is available.")

    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚡ Quick Stats</div>', unsafe_allow_html=True)
    
    if latest_data:
        # Quick environmental assessment
        temp_status = "Optimal" if 18 <= temperature <= 28 else "Check"
        moisture_status = "Good" if 40 <= soil_moisture <= 80 else "Monitor"
        light_status = "Adequate" if light_intensity >= 300 else "Low"
        
        st.markdown("""
        <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #22c55e;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Temperature:</span>
                <strong>{}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Soil Moisture:</span>
                <strong>{}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Light Levels:</span>
                <strong>{}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Overall:</span>
                <strong>Stable</strong>
            </div>
        </div>
        """.format(temp_status, moisture_status, light_status), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="status-text" style="margin-top: 1rem;">
            💡 Tip: Maintain soil moisture between 40-80% for optimal tomato growth
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Quick stats will appear when sensor data is available.")
    
    st.markdown("</div>", unsafe_allow_html=True)
