import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import os
import pandas as pd
from cryptography.fernet import Fernet

# =====================================================
# Config - Force Light Theme with Green Colors
# =====================================================
st.set_page_config(
    page_title="TomoGrow – Smart Irrigation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f8fdf8 !important; }
    .main .block-container {
        background-color: #f8fdf8 !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stMarkdown, .stText, .stWrite, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #1a331c !important;
    }
    [data-testid="metric-container"] { background-color: transparent !important; }
    [data-testid="metric-container"] label,
    [data-testid="metric-container"] div {
        color: #1a331c !important;
    }
    .dataframe {
        background-color: white !important;
        color: #1a331c !important;
    }
    .stSlider, .stSelectbox {
        background-color: transparent !important;
        color: #1a331c !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] {
        background: #22c55e !important;
        height: 4px !important;
        border-radius: 999px !important;
    }
    div.stSlider > div[data-baseweb="slider"] {
        background: transparent !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }
    div.stSlider [role="slider"] {
        background-color: #ef4444 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.35) !important;
    }
    .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #f0f8f0 !important;
        color: #1a331c !important;
        border-left: 4px solid #22c55e;
    }
    .stProgress > div > div { background-color: #22c55e; }
    .stButton button { background-color: #22c55e; color: white; }
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
    .card-title::before { content: "🌿"; font-size: 1.3em; }
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
    .status-text {
        font-size: 0.85rem;
        color: #4d7c0f;
        font-style: italic;
        background: #f0fdf4;
        padding: 0.5rem;
        border-radius: 6px;
        border-left: 3px solid #22c55e;
    }
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
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #22c55e 50%, transparent 100%);
        margin: 2rem 0;
        opacity: 0.3;
    }
    .control-section {
        background: #f8fdf8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dcfce7;
        margin-bottom: 1.5rem;
    }
    .simulation-controls {
        background: linear-gradient(135deg, #f0fdf4 0%, #e6f7ed 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #bbf7d0;
        margin-bottom: 1rem;
    }
    .current-values {
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
        margin-top: 1rem;
    }
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.1);
        border: 1px solid #dcfce7;
    }
    .login-title {
        text-align: center;
        color: #166534;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Supabase + Encryption config
# =====================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
DEVICE_ID = "ESP32_TOMOGROW_001"

DATA_ENCRYPTION_KEY = st.secrets["DATA_ENCRYPTION_KEY"]
cipher = Fernet(DATA_ENCRYPTION_KEY)

def dec_number(s):
    if s is None:
        return None
    return float(cipher.decrypt(s.encode()).decode())

@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # Test connection by making a simple query
        response = client.table("profiles").select("*").limit(1).execute()
        st.success("✅ Connected to Supabase successfully")
        return client
    except Exception as e:
        st.error(f"❌ Supabase connection error: {e}")
        st.info("Please check your Supabase URL and Anon Key in secrets.toml")
        return None

supabase_client = init_supabase()

# =====================================================
# Authentication
# =====================================================
def ensure_auth():
    if "user" in st.session_state:
        return

    # First, ensure Supabase client is initialized
    if supabase_client is None:
        st.error("Database connection failed. Cannot authenticate.")
        st.stop()

    # Display login form
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🌱 TomoGrow Login</div>', unsafe_allow_html=True)
    
    email = st.text_input("Email", value="soundous@gmail.com", placeholder="Enter your email")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Sign In", type="primary", use_container_width=True):
            # Validate inputs
            if not email or not password:
                st.error("Please enter both email and password")
                st.stop()
            
            try:
                # Use the correct auth method signature
                auth_response = supabase_client.auth.sign_in_with_password({
                    "email": email.strip(),
                    "password": password.strip()
                })
                
                if auth_response.user:
                    st.session_state["user"] = auth_response.user
                    st.session_state["user_email"] = auth_response.user.email
                    st.success(f"Welcome, {auth_response.user.email}!")
                    st.rerun()
                else:
                    st.error("Login failed. Please check your credentials.")
                    
            except Exception as e:
                # Provide more detailed error information
                error_msg = str(e)
                if "Invalid login credentials" in error_msg:
                    st.error("Invalid email or password. Please try again.")
                elif "Email not confirmed" in error_msg:
                    st.error("Please confirm your email address before logging in.")
                elif "rate limit" in error_msg.lower():
                    st.error("Too many login attempts. Please try again later.")
                else:
                    st.error(f"Login error: {error_msg}")
                
                # Debug information
                with st.expander("Troubleshooting Tips"):
                    st.write("""
                    1. Check if the user exists in Supabase Auth → Users
                    2. Verify email is confirmed
                    3. Check your internet connection
                    4. Ensure Supabase project is running
                    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Don't proceed if not authenticated
    st.stop()

# Call authentication function
ensure_auth()

# Now we have a user, proceed
current_user = st.session_state["user"]

# =====================================================
# Role loading from profiles
# =====================================================
@st.cache_data(show_spinner=False)
def get_current_role(user_id: str):
    try:
        res = (
            supabase_client
            .table("profiles")
            .select("role")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if res.data is None:
            return "farmer"
        return res.data.get("role", "farmer")
    except Exception:
        return "farmer"

if "role" not in st.session_state:
    st.session_state["role"] = get_current_role(current_user.id)

role = st.session_state["role"]

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
# Data access (decrypting)
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
            # Decrypt columns
            if "temperature" in df.columns:
                df["temperature"] = df["temperature"].apply(dec_number)
            if "humidity" in df.columns:
                df["humidity"] = df["humidity"].apply(dec_number)
            if "soil_moisture" in df.columns:
                df["soil_moisture"] = df["soil_moisture"].apply(dec_number)
            if "light_intensity" in df.columns:
                df["light_intensity"] = df["light_intensity"].apply(dec_number)
            df = df.sort_values("created_at")
            return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return None

# =====================================================
# Header
# =====================================================
st.markdown(
    f"""
    <div class="header">
        <div class="header-title">🌱 TomoGrow – Smart Irrigation Monitor</div>
        <div class="header-subtitle">
            Welcome, {current_user.email} | Role: {role.capitalize()} | Device: {DEVICE_ID}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# FARMER DASHBOARD
# =====================================================
def render_farmer_dashboard():
    latest_data = get_latest_data()
    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Live Field Snapshot</div>', unsafe_allow_html=True)

        if latest_data:
            temperature = dec_number(latest_data.get("temperature"))
            humidity = dec_number(latest_data.get("humidity"))
            soil_moisture = dec_number(latest_data.get("soil_moisture"))
            light_intensity = dec_number(latest_data.get("light_intensity"))
            timestamp = latest_data.get("created_at", "")

            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🌡️ Temperature</div>
                    <div class="metric-value">{temperature}°C</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💧 Humidity</div>
                    <div class="metric-value">{humidity}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🌱 Soil Moisture</div>
                    <div class="metric-value">{soil_moisture}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">☀️ Light</div>
                    <div class="metric-value">{int(light_intensity)}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
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
            temperature = dec_number(latest_data.get("temperature"))
            humidity = dec_number(latest_data.get("humidity"))
            soil_moisture = dec_number(latest_data.get("soil_moisture"))
            light_intensity = dec_number(latest_data.get("light_intensity"))
            
            result = predict_irrigation_model_only(temperature, soil_moisture, humidity, light_intensity)
            if result:
                decision = result["irrigation_prediction"]
                conf = result["confidence_level"]

                if decision == "yes":
                    st.success("💦 Water the plants now")
                    st.write("Current conditions suggest watering would benefit the plants for optimal growth.")
                    st.progress(conf)
                else:
                    st.info("✅ No water needed")
                    st.write("Conditions are comfortable for the plants. Continue monitoring.")
                    st.progress(conf)
                st.write(f"**Confidence Level:** {conf:.0%}")
            else:
                st.warning("⚠️ Unable to generate irrigation advice at this time.")
        else:
            st.info("⏳ Waiting for data and model to generate irrigation advice.")

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Sensor History & Trends</div>', unsafe_allow_html=True)

        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.write("**Chart Configuration**")
        control_col1, control_col2 = st.columns(2)

        with control_col1:
            points = st.slider(
                "Data points to display",
                min_value=20,
                max_value=200,
                value=80,
                step=20,
                help="Number of historical data points to show on the chart"
            )

        with control_col2:
            metric_choice = st.selectbox(
                "Select metric",
                ["temperature", "humidity", "soil_moisture", "light_intensity"],
                index=2,
                format_func=lambda x: {
                    "temperature": "🌡️ Temperature",
                    "humidity": "💧 Humidity",
                    "soil_moisture": "🌱 Soil Moisture",
                    "light_intensity": "☀️ Light Intensity"
                }[x]
            )

        st.markdown('</div>', unsafe_allow_html=True)

        df_hist = get_history(limit=points)
        if df_hist is not None:
            st.markdown("**Live Trend**")
            st.line_chart(
                df_hist.set_index("created_at")[metric_choice],
                height=320
            )
            st.markdown("**Recent Measurements**")
            st.dataframe(
                df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("📊 No historical data available yet. Data will accumulate over time.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # SIMULATION SECTION
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔬 Simulation Lab</div>', unsafe_allow_html=True)
    st.markdown('<p class="status-text">Test how different environmental conditions affect irrigation needs</p>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1.2, 1.2])
    sim_result = None

    with sim_col1:
        st.markdown('<div class="simulation-controls">', unsafe_allow_html=True)
        st.write("**Adjust environmental parameters:**")

        sim_temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0, 0.5, key="sim_temp")
        sim_soil = st.slider("💧 Soil Moisture (%)", 0.0, 100.0, 50.0, 1.0, key="sim_soil")
        sim_hum = st.slider("🌫️ Air Humidity (%)", 0.0, 100.0, 60.0, 1.0, key="sim_hum")
        sim_light = st.slider("☀️ Light Intensity", 0, 1500, 500, 10, key="sim_light")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
            <div class="current-values">
                <div style="text-align: center; margin-bottom: 0.5rem; font-weight: 600; color: #166534;">Current Simulation Values</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #4d7c0f;">Temperature</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{sim_temp}°C</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #4d7c0f;">Soil Moisture</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{sim_soil}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #4d7c0f;">Humidity</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{sim_hum}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: #4d7c0f;">Light</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{sim_light}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
                    st.success(f"💦 Simulated Advice: Water Recommended")
                    st.write(f"With these conditions, the model suggests watering with **{sim_conf:.0%} confidence**")
                else:
                    st.info(f"✅ Simulated Advice: No Water Needed")
                    st.write(f"Current simulated conditions don't require watering (**{sim_conf:.0%} confidence**)")

    with sim_col2:
        st.markdown('<div class="simulation-controls">', unsafe_allow_html=True)
        st.write("🌿 Simulated Plant Response")

        if artifacts is not None and sim_result is not None:
            sim_decision = sim_result["irrigation_prediction"]

            if sim_soil > 70 and sim_decision == "no":
                sim_state_label = "Thriving"
                sim_emoji = "🌿"
                sim_note = "Perfect conditions! The plant would be lush and vibrant with optimal soil moisture."
                sim_status_class = "plant-status-healthy"
            elif sim_soil < 40 or sim_decision == "yes":
                sim_state_label = "Stressed"
                sim_emoji = "🥀"
                sim_note = "The plant would show signs of dehydration. Leaves might droop and soil feels dry."
                sim_status_class = "plant-status-attention"
            else:
                sim_state_label = "Stable"
                sim_emoji = "🌱"
                sim_note = "The plant would be growing steadily but could benefit from improved conditions."
                sim_status_class = "plant-status-stable"

            st.markdown(f"""
                <div class="metric-card {sim_status_class}" style="text-align: center; padding: 1.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{sim_emoji}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; color: inherit;">{sim_state_label}</div>
                    <div style="font-size: 0.9rem; color: inherit;">{sim_note}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("**Simulated Environment:**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Soil", f"{sim_soil}%")
                st.metric("Light", f"{sim_light}")
            with c2:
                st.metric("Temp", f"{sim_temp}°C")
                st.metric("Humidity", f"{sim_hum}%")
        else:
            st.info("🎛️ Adjust the sliders on the left to see how different conditions affect plant health and irrigation needs.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ADMIN DASHBOARD
# =====================================================
def render_admin_dashboard():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👩‍💻 Admin Panel – Users & Devices</div>', unsafe_allow_html=True)
    st.write("Simple admin view of recent sensor data for all devices/users.")

    try:
        res = (
            supabase_client
            .table("sensor_data")
            .select("id, device_id, user_id, temperature, soil_moisture, created_at")
            .order("id", desc=True)
            .limit(50)
            .execute()
        )
        df = pd.DataFrame(res.data or [])
        if not df.empty:
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No sensor data yet.")
    except Exception as e:
        st.error(f"Error loading admin data: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Logout button
# =====================================================
with st.sidebar:
    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =====================================================
# ROUTER
# =====================================================
if role == "admin":
    render_admin_dashboard()
else:
    render_farmer_dashboard()
