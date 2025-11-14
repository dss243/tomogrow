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
    layout="wide",
    initial_sidebar_state="collapsed"
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
# Enhanced Styling – Modern & Beautiful
# =====================================================
st.markdown(
    """
    <style>
    /* Modern color palette */
    :root {
        --primary: #10b981;
        --primary-light: #d1fae5;
        --primary-dark: #047857;
        --secondary: #f0fdf4;
        --accent: #059669;
        --text: #1f2937;
        --text-light: #6b7280;
        --background: #f8fafc;
        --card: #ffffff;
        --border: #e5e7eb;
    }
    
    body {
        background: linear-gradient(135deg, #f0fdf4 0%, #f8fafc 50%, #ecfdf5 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 100%;
    }
    
    /* Header with gradient */
    .title-box {
        padding: 1.5rem 2rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        border: none;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .title-main {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
        letter-spacing: -0.02em;
    }
    
    .title-sub {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Modern cards with subtle shadows */
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: var(--card);
        border: 1px solid var(--border);
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-title::before {
        content: "🌱";
        font-size: 1.2em;
    }
    
    /* Enhanced metric boxes */
    .metric-box {
        padding: 0.8rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, var(--primary-light) 0%, #f0fdf4 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        text-align: center;
    }
    
    .metric-box .stMetric {
        background: transparent !important;
    }
    
    /* Plant status with animations */
    .plant-container {
        text-align: center;
        padding: 1rem;
    }
    
    .plant-state {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    
    .plant-emoji {
        font-size: 4rem;
        margin: 0.5rem 0;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .plant-note {
        font-size: 0.95rem;
        color: var(--text-light);
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-happy {
        color: var(--primary-dark);
    }
    
    .status-thirsty {
        color: #dc2626;
    }
    
    .status-tired {
        color: #d97706;
    }
    
    /* Enhanced typography */
    .small-muted {
        font-size: 0.8rem;
        color: var(--text-light);
        font-style: italic;
    }
    
    /* Custom slider styling */
    .stSlider > div > div {
        background: var(--primary-light);
    }
    
    /* Button and interactive elements */
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        border: none;
        border-radius: 0.7rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="title-box">
        <div class="title-main">🌿 TomoGrow – Smart Irrigation Monitor</div>
        <div class="title-sub">
            Real-time insights for healthier plants and smarter watering decisions
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Layout: top = live + advice + plant, bottom = history + simulation
# =====================================================
latest_data = get_latest_data()

top_left, top_right = st.columns([1.4, 1.6])

# ---------------------- TOP LEFT: LIVE + ADVICE + PLANT ----------------------
with top_left:
    # Live snapshot
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Live Field Snapshot</div>', unsafe_allow_html=True)

    result_for_plant = None
    temperature = humidity = soil_moisture = light_intensity = None

    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("🌡️ Temperature", f"{temperature:.1f}°C")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("💧 Humidity", f"{humidity:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("🌱 Soil Moisture", f"{soil_moisture:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("☀️ Light", f"{int(light_intensity)}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<p class="small-muted">📅 Last update from the field: {timestamp}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("🔍 No sensor data yet. When the device starts sending, values will appear here.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Irrigation advice
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💧 Irrigation Advice</div>', unsafe_allow_html=True)

    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(
            temperature,
            soil_moisture,
            humidity,
            light_intensity,
        )

        result_for_plant = result

        if result is None:
            st.warning("🔄 The system is not ready to give advice yet.")
        else:
            decision = result["irrigation_prediction"]
            conf = result["confidence_level"]

            if decision == "yes":
                st.success("💦 **Advice: Water the plants now**")
                st.write("Soil and weather conditions suggest that watering would help the plants thrive.")
            else:
                st.info("✅ **Advice: No water needed at the moment**")
                st.write("Current conditions look comfortable; watering can wait.")

            # Confidence indicator
            col_conf, col_bar = st.columns([1, 3])
            with col_conf:
                st.metric("Confidence", f"{conf:.0%}")
            with col_bar:
                st.progress(conf)
            
            st.markdown(
                '<p class="small-muted">🎯 Based on soil moisture, air temperature, humidity and light intensity analysis</p>',
                unsafe_allow_html=True,
            )
    else:
        st.info("⏳ Waiting for live data and the irrigation advisor to start.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Plant view (emoji)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌿 Plant Health Status</div>', unsafe_allow_html=True)

    if latest_data and artifacts is not None and result_for_plant is not None:
        decision = result_for_plant["irrigation_prediction"]

        if soil_moisture is not None and soil_moisture > 70 and decision == "no":
            state_label = "Happy & Healthy"
            plant_emoji = "🌿"
            note = "Leaves look firm and vibrant. Soil moisture is optimal. Your plant is thriving!"
            status_class = "status-happy"
        elif soil_moisture is not None and (soil_moisture < 40 or decision == "yes"):
            state_label = "Needs Water"
            plant_emoji = "🥀"
            note = "The plant is showing signs of stress. Soil is drying out and immediate watering is recommended."
            status_class = "status-thirsty"
        else:
            state_label = "Doing Okay"
            plant_emoji = "🌱"
            note = "The plant is stable but not at peak condition. Monitor closely for changes."
            status_class = "status-tired"

        st.markdown(f'<div class="plant-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-state {status_class}">{state_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-emoji">{plant_emoji}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-note">{note}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("🌱 When live data arrives, this will show real-time plant health status.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- TOP RIGHT: HISTORY ----------------------
with top_right:
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

    if df_hist is None:
        st.info("📊 No historical data yet. Data will appear as the system collects more readings.")
    else:
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

        # Enhanced chart
        chart_data = df_hist.set_index("created_at")[metric_choice]
        st.line_chart(
            chart_data,
            height=350,
            use_container_width=True
        )

        st.markdown(
            '<p class="small-muted">💡 Smooth curves indicate stable conditions, while spikes may show irrigation events or weather changes</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**Recent Measurements**")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(8),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- BOTTOM: SIMULATION SECTION ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔬 Simulation Lab</div>', unsafe_allow_html=True)
st.markdown('<p class="small-muted">Test how different environmental conditions affect irrigation needs</p>', unsafe_allow_html=True)

col_sim1, col_sim2 = st.columns([1.2, 1.2])

with col_sim1:
    st.write("**Adjust environmental parameters:**")

    sim_temp = st.slider("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5, 
                        help="Higher temperatures increase water evaporation")
    sim_soil = st.slider("💧 Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                        help="Lower values indicate drier soil needing water")
    sim_hum = st.slider("🌫️ Air Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0,
                       help="Higher humidity reduces water loss from plants")
    sim_light = st.slider("☀️ Light Intensity", min_value=0, max_value=1500, value=500, step=10,
                         help="More light increases photosynthesis and water usage")

    if artifacts is None:
        st.warning("🤖 The AI model is not loaded. Simulation features are currently unavailable.")
    else:
        sim_result = model_predict(sim_temp, sim_soil, sim_hum, sim_light, crop_type="tomato")

        if sim_result is None:
            st.error("❌ Could not compute simulation with these values.")
        else:
            sim_decision = sim_result["irrigation_prediction"]
            sim_conf = sim_result["confidence_level"]

            if sim_decision == "yes":
                st.success(f"💦 **Simulated Advice: Water Recommended**")
                st.write(f"With these conditions, the model suggests watering with **{sim_conf:.0%} confidence**")
            else:
                st.info(f"✅ **Simulated Advice: No Water Needed**")
                st.write(f"Current simulated conditions don't require watering (**{sim_conf:.0%} confidence**)")

with col_sim2:
    st.write("**🌿 Simulated Plant Response**")

    if artifacts is not None and sim_result is not None:
        sim_decision = sim_result["irrigation_prediction"]
        
        if sim_soil > 70 and sim_decision == "no":
            sim_state_label = "Thriving"
            sim_emoji = "🌿"
            sim_note = "Perfect conditions! The plant would be lush and vibrant with optimal soil moisture."
            sim_status_class = "status-happy"
        elif sim_soil < 40 or sim_decision == "yes":
            sim_state_label = "Stressed"
            sim_emoji = "🥀"
            sim_note = "The plant would show signs of dehydration. Leaves might droop and soil feels dry."
            sim_status_class = "status-thirsty"
        else:
            sim_state_label = "Stable"
            sim_emoji = "🌱"
            sim_note = "The plant would be growing steadily but could benefit from improved conditions."
            sim_status_class = "status-tired"

        st.markdown(f'<div class="plant-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-state {sim_status_class}">{sim_state_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-emoji">{sim_emoji}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-note">{sim_note}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("**Simulated Environment:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Soil", f"{sim_soil}%")
            st.metric("Light", f"{sim_light}")
        with col2:
            st.metric("Temp", f"{sim_temp}°C")
            st.metric("Humidity", f"{sim_hum}%")
    else:
        st.info("🎛️ Adjust the sliders to see how different conditions affect plant health and irrigation needs.")

st.markdown("</div>", unsafe_allow_html=True)
