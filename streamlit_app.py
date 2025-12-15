import os
import json
import time
import hmac
import hashlib
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from supabase import create_client
from cryptography.fernet import Fernet
import pickle
import joblib
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG & STYLES
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
    .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #f0f8f0 !important;
        color: #1a331c !important;
        border-left: 4px solid #22c55e;
    }
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
    }
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
# CONFIG & SECRETS
# =====================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
DEVICE_ID = "ESP32_TOMOGROW_001"

DATA_ENCRYPTION_KEY = st.secrets["DATA_ENCRYPTION_KEY"]
cipher = Fernet(DATA_ENCRYPTION_KEY)

MODEL_SIGNING_KEY_HEX = st.secrets["MODEL_SIGNING_KEY"]
MODEL_SIGNING_KEY = bytes.fromhex(MODEL_SIGNING_KEY_HEX)

MODEL_DIR = "models"

CLASS_MODEL_FILE = "fast_tomato_irrigation_model.pkl"
WATER_MODEL_FILE = "water_volume_model_final.keras"
WATER_PREP_FILE = "water_preprocessing_final.pkl"
SIGNATURE_FILE = "model_signatures.json"

MODEL_INFO = {
    "classification": "rf_irrigation_v1.0",
    "water_volume": "water_nn_v1.0",
    "last_validation": "2025-12-10",
}

# =====================================================
# UTILS: DECRYPTION & HASH / SIGNATURE
# =====================================================
def dec_number(s):
    if s is None:
        return None
    try:
        return float(cipher.decrypt(s.encode()).decode())
    except Exception:
        return None

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@st.cache_resource
def verify_model_signatures():
    sig_path = os.path.join(MODEL_DIR, SIGNATURE_FILE)
    if not os.path.exists(sig_path):
        st.error("Model signature file missing. Cannot verify model integrity.")
        st.stop()

    with open(sig_path, "r") as f:
        ref = json.load(f)

    for fname, info in ref.items():
        full_path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(full_path):
            st.error(f"Model file {fname} not found.")
            st.stop()

        digest = file_sha256(full_path)
        if digest != info["sha256"]:
            st.error(f"Hash mismatch for {fname}. Possible tampering.")
            st.stop()

        expected_hmac = info["hmac"]
        actual_hmac = hmac.new(MODEL_SIGNING_KEY, digest.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_hmac, actual_hmac):
            st.error(f"Signature verification failed for {fname}.")
            st.stop()

    st.success("✅ Model signatures verified")
    return True

# =====================================================
# SUPABASE INIT
# =====================================================
@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        client.table("profiles").select("*").limit(1).execute()
        st.success("✅ Connected to Supabase successfully")
        return client
    except Exception as e:
        st.error(f"❌ Supabase connection error: {e}")
        st.stop()

supabase_client = init_supabase()

# =====================================================
# AUTHENTICATION
# =====================================================
def ensure_auth():
    if "user" in st.session_state:
        return

    if supabase_client is None:
        st.error("Database connection failed. Cannot authenticate.")
        st.stop()

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🌱 TomoGrow Login</div>', unsafe_allow_html=True)

    email = st.text_input("Email", value="", placeholder="Enter your email")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Sign In", type="primary", use_container_width=True):
            if not email or not password:
                st.error("Please enter both email and password")
                st.stop()

            try:
                auth_response = supabase_client.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password.strip()}
                )
                if auth_response.user:
                    st.session_state["user"] = auth_response.user
                    st.session_state["user_email"] = auth_response.user.email
                    st.success(f"Welcome, {auth_response.user.email}!")
                    st.rerun()
                else:
                    st.error("Login failed. Please check your credentials.")
            except Exception as e:
                st.error(f"Login error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

ensure_auth()
current_user = st.session_state["user"]

# =====================================================
# ROLE LOADING
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
# VERIFY MODEL SIGNATURES FIRST
# =====================================================
verify_model_signatures()

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_classification_artifacts():
    path = os.path.join(MODEL_DIR, CLASS_MODEL_FILE)
    if not os.path.exists(path):
        st.error("Classification model file not found.")
        return None
    try:
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        # Expect keys: model, scaler, crop_encoder, pump_encoder, feature_names
        return artifacts
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

@st.cache_resource
def load_water_volume_artifacts():
    model_path = os.path.join(MODEL_DIR, WATER_MODEL_FILE)
    prep_path = os.path.join(MODEL_DIR, WATER_PREP_FILE)
    if not os.path.exists(model_path):
        st.warning("Water volume model file not found.")
        return None
    if not os.path.exists(prep_path):
        st.warning("Water preprocessing file not found.")
        return None
    try:
        water_model = load_model(model_path)
        preprocessing = joblib.load(prep_path)
        return {
            "model": water_model,
            "feature_scaler": preprocessing["feature_scaler"],
            "target_scaler": preprocessing["target_scaler"],
            "feature_names": preprocessing["feature_names"],
        }
    except Exception as e:
        st.error(f"Error loading water volume artifacts: {e}")
        return None

class_artifacts = load_classification_artifacts()
water_artifacts = load_water_volume_artifacts()

def model_predict_class(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    if class_artifacts is None:
        return None
    model = class_artifacts["model"]
    scaler = class_artifacts["scaler"]
    crop_encoder = class_artifacts["crop_encoder"]
    pump_encoder = class_artifacts["pump_encoder"]

    try:
        crop_code = crop_encoder.transform([crop_type])[0]
    except Exception:
        crop_code = 0

    features = np.array([[float(temperature),
                          float(soil_moisture),
                          float(humidity),
                          float(light_intensity),
                          crop_code]])
    try:
        features_scaled = scaler.transform(features)
        prediction_encoded = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)[0]
    except Exception as e:
        st.error(f"Error during classification prediction: {e}")
        return None

    label = pump_encoder.inverse_transform([prediction_encoded])[0]
    conf = float(probs[prediction_encoded])
    return {
        "irrigation_prediction": label,
        "confidence": min(conf, 0.95),
        "probabilities": {
            "no": float(probs[0]),
            "yes": float(probs[1]),
        },
    }

def predict_water_volume(soil_moisture, soil_temperature, soil_humidity, ts=None):
    if water_artifacts is None:
        return None

    model = water_artifacts["model"]
    scaler_X = water_artifacts["feature_scaler"]
    scaler_y = water_artifacts["target_scaler"]
    feature_names = water_artifacts["feature_names"]

    if ts is None:
        ts = datetime.utcnow()
    if not isinstance(ts, datetime):
        ts = pd.to_datetime(ts)

    hour = ts.hour
    day_of_week = ts.weekday()
    month = ts.month
    day_of_year = ts.timetuple().tm_yday

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    temp_humidity_interaction = soil_temperature * soil_humidity
    temp_moisture_interaction = soil_temperature * soil_moisture

    feature_dict = {
        "soil_moisture": soil_moisture,
        "soil_temperature": soil_temperature,
        "soil_humidity": soil_humidity,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "day_of_year": day_of_year,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "temp_humidity_interaction": temp_humidity_interaction,
        "temp_moisture_interaction": temp_moisture_interaction,
    }

    x_list = [feature_dict.get(name, 0.0) for name in feature_names]
    X_input = np.array(x_list, dtype=np.float32).reshape(1, -1)
    try:
        X_scaled = scaler_X.transform(X_input)
        y_scaled = model.predict(X_scaled, verbose=0).flatten()[0]
        y_pred = scaler_y.inverse_transform([[y_scaled]])[0][0]
        return max(float(y_pred), 0.0)
    except Exception as e:
        st.error(f"Error during water volume prediction: {e}")
        return None

def estimate_soil_moisture_after_watering(current_sm, volume_ml):
    if volume_ml is None:
        return current_sm
    delta = volume_ml / 1000.0
    new_sm = current_sm + delta
    return float(max(0.0, min(100.0, new_sm)))

# =====================================================
# DATA ACCESS (SUPABASE)
# =====================================================
def get_latest_data():
    try:
        resp = (
            supabase_client
            .table("sensor_data")
            .select("*")
            .eq("device_id", DEVICE_ID)
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
    return None

def get_history(limit=100):
    try:
        resp = (
            supabase_client
            .table("sensor_data")
            .select("*")
            .eq("device_id", DEVICE_ID)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        data = resp.data or []
        if not data:
            return None
        df = pd.DataFrame(data)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
        for col in ["temperature", "humidity", "soil_moisture", "light_intensity"]:
            if col in df.columns:
                df[col] = df[col].apply(dec_number)
        df = df.sort_values("created_at")
        return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return None

# =====================================================
# SIMPLE INPUT VALIDATION
# =====================================================
def is_plausible_reading(temp, hum, soil, light):
    if temp is None or hum is None or soil is None or light is None:
        return False
    if not (-5 <= temp <= 60):
        return False
    if not (0 <= hum <= 100):
        return False
    if not (0 <= soil <= 100):
        return False
    if not (0 <= light <= 2000):
        return False
    return True

def is_reasonable_jump(prev_val, curr_val, max_delta):
    if prev_val is None or curr_val is None:
        return True
    return abs(curr_val - prev_val) <= max_delta

def can_request_prediction():
    now = time.time()
    last = st.session_state.get("last_pred_time", 0)
    if now - last < 5:
        return False
    st.session_state["last_pred_time"] = now
    return True

# =====================================================
# HEADER
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

st.sidebar.markdown(
    f"**Model status**  \n"
    f"- Classifier: {MODEL_INFO['classification']}  \n"
    f"- Volume: {MODEL_INFO['water_volume']}  \n"
    f"- Last validation: {MODEL_INFO['last_validation']}"
)

# =====================================================
# DASHBOARDS
# =====================================================
def render_farmer_dashboard():
    latest_data = get_latest_data()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Live Field Snapshot", unsafe_allow_html=True)

        if latest_data:
            temperature = dec_number(latest_data.get("temperature"))
            humidity = dec_number(latest_data.get("humidity"))
            soil_moisture = dec_number(latest_data.get("soil_moisture"))
            light_intensity = dec_number(latest_data.get("light_intensity"))
            timestamp = latest_data.get("created_at", "")

            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🌡️ Temperature</div>'
                f'<div class="metric-value">{temperature}°C</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">💧 Humidity</div>'
                f'<div class="metric-value">{humidity}%</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🌱 Soil Moisture</div>'
                f'<div class="metric-value">{soil_moisture}%</div></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">☀️ Light</div>'
                f'<div class="metric-value">{int(light_intensity)}</div></div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-text">🕐 Last update from the field: {timestamp}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("📡 No sensor data available.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Irrigation Advice
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💧 Irrigation Advice", unsafe_allow_html=True)

        if latest_data and class_artifacts is not None:
            temperature = dec_number(latest_data.get("temperature"))
            humidity = dec_number(latest_data.get("humidity"))
            soil_moisture = dec_number(latest_data.get("soil_moisture"))
            light_intensity = dec_number(latest_data.get("light_intensity"))
            ts = latest_data.get("created_at")

            valid = is_plausible_reading(temperature, humidity, soil_moisture, light_intensity)
            df_hist = get_history(limit=5)
            suspicious = False
            if df_hist is not None and len(df_hist) >= 2:
                last2 = df_hist.tail(2)
                prev = last2.iloc[0]
                curr = last2.iloc[1]
                if not is_reasonable_jump(prev["soil_moisture"], curr["soil_moisture"], 30):
                    suspicious = True
                if not is_reasonable_jump(prev["temperature"], curr["temperature"], 10):
                    suspicious = True

            if not can_request_prediction():
                st.info("⏳ Please wait a few seconds before a new prediction.")
            elif not valid or suspicious:
                st.warning("⚠️ Sensor data look atypical. AI recommendation disabled for safety.")
            else:
                res = model_predict_class(temperature, soil_moisture, humidity, light_intensity)
                if res:
                    decision = res["irrigation_prediction"]
                    conf = res["confidence"]
                    if decision == "yes":
                        st.success("💦 Water the plants now")
                        st.write("Current conditions suggest watering would benefit the plants.")
                        vol_ml = None
                        if water_artifacts is not None:
                            vol_ml = predict_water_volume(
                                soil_moisture=soil_moisture,
                                soil_temperature=temperature,
                                soil_humidity=humidity,
                                ts=ts,
                            )
                            if vol_ml is not None:
                                st.metric("Recommended water volume", f"{vol_ml:,.0f} ml")
                        st.progress(conf)
                    else:
                        st.info("✅ No water needed")
                        st.write("Conditions are comfortable for the plants.")
                        st.progress(conf)
                    st.write(f"**Confidence Level:** {conf:.0%}")
                else:
                    st.warning("⚠️ Unable to generate irrigation advice.")
        else:
            st.info("⏳ Waiting for data/models to generate advice.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Sensor History & Trends", unsafe_allow_html=True)

        df_hist = get_history(limit=200)
        if df_hist is not None:
            st.write("**Chart Configuration**")
            c1, c2 = st.columns(2)
            with c1:
                points = st.slider("Data points to display", 20, 200, 80, 20)
            with c2:
                metric_choice = st.selectbox(
                    "Select metric",
                    ["temperature", "humidity", "soil_moisture", "light_intensity"],
                    index=2,
                )

            df_sel = df_hist.tail(points)
            st.markdown("**Live Trend**")
            st.line_chart(df_sel.set_index("created_at")[metric_choice], height=320)
            st.markdown("**Recent Measurements**")
            st.dataframe(
                df_sel[["created_at", "temperature", "humidity",
                        "soil_moisture", "light_intensity"]].tail(6),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("📊 No historical data yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Simulation Lab (short version)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Simulation Lab", unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        sim_temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
        sim_soil = st.slider("💧 Soil Moisture (%)", 0.0, 100.0, 50.0, 1.0)
        sim_hum = st.slider("🌫️ Air Humidity (%)", 0.0, 100.0, 60.0, 1.0)
        sim_light = st.slider("☀️ Light Intensity", 0, 1500, 500, 10)

        if class_artifacts is None:
            st.warning("🤖 Classification model not loaded.")
        else:
            res = model_predict_class(sim_temp, sim_soil, sim_hum, sim_light)
            if res:
                decision = res["irrigation_prediction"]
                conf = res["confidence"]
                if decision == "yes":
                    st.success(f"💦 Simulated Advice: Water Recommended ({conf:.0%})")
                    sim_vol = None
                    if water_artifacts is not None:
                        sim_vol = predict_water_volume(
                            soil_moisture=sim_soil,
                            soil_temperature=sim_temp,
                            soil_humidity=sim_hum,
                            ts=datetime.utcnow(),
                        )
                        if sim_vol is not None:
                            st.metric("Simulated water volume", f"{sim_vol:,.0f} ml")
                    est_sm = estimate_soil_moisture_after_watering(sim_soil, sim_vol)
                    st.write(f"Estimated soil moisture after irrigation: **{est_sm:.1f}%**")
                else:
                    st.info(f"✅ Simulated Advice: No Water Needed ({conf:.0%})")

    with sim_c2:
        st.write("🌿 Simulated Plant Response")
        if class_artifacts is not None:
            if sim_soil > 70:
                st.success("Plant state: Thriving")
            elif sim_soil < 40:
                st.warning("Plant state: Stressed")
            else:
                st.info("Plant state: Stable")

    st.markdown("</div>", unsafe_allow_html=True)

def render_admin_dashboard():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👩‍💻 Admin Panel – Users & Devices", unsafe_allow_html=True)
    st.write("Recent sensor data (encrypted fields stored in DB).")

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
# SIDEBAR LOGOUT
# =====================================================
with st.sidebar:
    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =====================================================
# ROUTING
# =====================================================
if role == "admin":
    render_admin_dashboard()
else:
    render_farmer_dashboard()
