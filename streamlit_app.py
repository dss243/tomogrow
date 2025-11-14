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
# Styling – light green, simple cards + emoji plant
# =====================================================
st.markdown(
    """
    <style>
    body {
        background-color: #f5faf5;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    .title-box {
        padding: 0.8rem 1.0rem;
        border-radius: 0.8rem;
        background: #f0f8f0;
        border: 1px solid #d6ead6;
        margin-bottom: 1.0rem;
    }
    .title-main {
        font-size: 2.0rem;
        font-weight: 650;
        margin: 0;
        color: #234221;
    }
    .title-sub {
        font-size: 0.95rem;
        color: #4f7a4c;
        margin-top: 0.25rem;
    }
    .card {
        padding: 0.9rem 1.0rem;
        border-radius: 0.8rem;
        background-color: #ffffff;
        border: 1px solid #e0ebe0;
        margin-bottom: 0.9rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #254024;
        margin-bottom: 0.4rem;
    }
    .metric-box {
        padding: 0.5rem 0.5rem;
        border-radius: 0.6rem;
        background-color: #f7fbf7;
        border: 1px solid #e1ece1;
    }
    .plant-state {
        font-size: 1.2rem;
        font-weight: 650;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .plant-emoji {
        font-size: 2.5rem;
        text-align: center;
        line-height: 1.1;
    }
    .plant-note {
        font-size: 0.9rem;
        color: #5f7a5f;
        text-align: center;
        margin-top: 0.3rem;
    }
    .small-muted {
        font-size: 0.8rem;
        color: #7b8b7b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="title-box">
        <div class="title-main">TomoGrow – Smart Irrigation Monitor</div>
        <div class="title-sub">
            A light overview of soil moisture, weather and when to water.
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
    st.markdown('<div class="card-title">Live field snapshot</div>', unsafe_allow_html=True)

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
            st.metric("Temperature", f"{temperature:.1f} °C")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Humidity", f"{humidity:.1f} %")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Soil moisture", f"{soil_moisture:.1f} %")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Light", f"{int(light_intensity)}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<p class="small-muted">Last update from the field: {timestamp}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.write("No sensor data yet. When the device starts sending, values will appear here.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Irrigation advice
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Irrigation advice</div>', unsafe_allow_html=True)

    if latest_data and artifacts is not None:
        result = predict_irrigation_model_only(
            temperature,
            soil_moisture,
            humidity,
            light_intensity,
        )

        result_for_plant = result

        if result is None:
            st.write("The system is not ready to give advice yet.")
        else:
            decision = result["irrigation_prediction"]
            conf = result["confidence_level"]

            if decision == "yes":
                st.write("Advice: water the plants now.")
                st.write("Soil and weather conditions suggest that watering would help the plants.")
            else:
                st.write("Advice: no water needed at the moment.")
                st.write("Current conditions look comfortable; watering can wait.")

            st.write(f"Confidence in this advice: about {conf:.0%}")
            st.markdown(
                '<p class="small-muted">Based on soil moisture, air temperature, humidity and light.</p>',
                unsafe_allow_html=True,
            )
    else:
        st.write("Waiting for live data and the irrigation advisor to start.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Plant view (emoji)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Plant view</div>', unsafe_allow_html=True)

    if latest_data and artifacts is not None and result_for_plant is not None:
        decision = result_for_plant["irrigation_prediction"]

        if soil_moisture is not None and soil_moisture > 70 and decision == "no":
            state_label = "Happy plant"
            plant_emoji = "🌿"
            note = "Leaves look firm. Soil feels moist. The plant is in a good mood."
        elif soil_moisture is not None and (soil_moisture < 40 or decision == "yes"):
            state_label = "Thirsty plant"
            plant_emoji = "🥀"
            note = "The plant is starting to droop. The soil is drying and water would help."
        else:
            state_label = "Tired plant"
            plant_emoji = "🌱"
            note = "The plant is not in danger, but it is not at its best. Conditions are average."

        st.markdown(f'<div class="plant-state">{state_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-emoji">{plant_emoji}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="plant-note">{note}</div>', unsafe_allow_html=True)
    else:
        st.write("When live data arrives, this box will show how the plant might feel.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- TOP RIGHT: HISTORY ----------------------
with top_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Sensor history</div>', unsafe_allow_html=True)

    points = st.slider(
        "Number of recent measurements",
        min_value=20,
        max_value=200,
        value=80,
        step=20,
    )

    df_hist = get_history(limit=points)

    if df_hist is None:
        st.write("No history yet. Leave the system running and readings will accumulate here.")
    else:
        metric_choice = st.selectbox(
            "Choose a variable to follow",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=2,
        )

        st.line_chart(
            df_hist.set_index("created_at")[metric_choice],
            height=320,
        )

        st.markdown(
            '<p class="small-muted">Slow curves often mean a relaxed field, while sudden peaks may be irrigation or weather events.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("Recent measurements")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- BOTTOM: SIMULATION SECTION ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Simulation: test different conditions</div>', unsafe_allow_html=True)

col_sim1, col_sim2 = st.columns([1.2, 1.2])

with col_sim1:
    st.write("Move the sliders to imagine a different moment in the field and see the advice change.")

    sim_temp = st.slider("Simulated temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    sim_soil = st.slider("Simulated soil moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    sim_hum = st.slider("Simulated air humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    sim_light = st.slider("Simulated light level", min_value=0, max_value=1500, value=500, step=10)

    if artifacts is None:
        st.write("The model is not loaded, simulation is not available.")
    else:
        sim_result = model_predict(sim_temp, sim_soil, sim_hum, sim_light, crop_type="tomato")

        if sim_result is None:
            st.write("Could not compute a simulation with these values.")
        else:
            sim_decision = sim_result["irrigation_prediction"]
            sim_conf = sim_result["confidence_level"]

            if sim_decision == "yes":
                st.write("Simulated advice: water the plants in this situation.")
            else:
                st.write("Simulated advice: no water needed in this situation.")

            st.write(f"Confidence in this simulated advice: about {sim_conf:.0%}")

with col_sim2:
    st.write("Simulated plant view")

    if artifacts is not None:
        if sim_result is not None:
            sim_decision = sim_result["irrigation_prediction"]
            # reuse same thresholds as live plant
            if sim_soil > 70 and sim_decision == "no":
                sim_state_label = "Happy plant"
                sim_emoji = "🌿"
                sim_note = "In this scenario, the soil is moist and the plant feels comfortable."
            elif sim_soil < 40 or sim_decision == "yes":
                sim_state_label = "Thirsty plant"
                sim_emoji = "🥀"
                sim_note = "In this scenario, the plant would like water."
            else:
                sim_state_label = "Tired plant"
                sim_emoji = "🌱"
                sim_note = "In this scenario, conditions are just okay, not ideal."

            st.markdown(f'<div class="plant-state">{sim_state_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="plant-emoji">{sim_emoji}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="plant-note">{sim_note}</div>', unsafe_allow_html=True)
        else:
            st.write("Move the sliders on the left to see a simulated plant state.")
    else:
        st.write("Model not loaded, plant simulation is not available.")

st.markdown("</div>", unsafe_allow_html=True)
