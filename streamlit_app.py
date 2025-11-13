import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import requests
import pickle
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Smart Irrigation Dashboard", page_icon="💧", layout="wide")
st.title("🌱 Smart Irrigation AI Dashboard")
st.markdown("---")

# ------------------- MODEL LOADING -------------------
@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    model_paths = [
        'fast_tomato_irrigation_model.pkl',
        './fast_tomato_irrigation_model.pkl',
        'model/fast_tomato_irrigation_model.pkl'
    ]
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                model_data = pickle.load(file)
            st.sidebar.success(f"✅ Model loaded from {path}")
            return model_data
    st.sidebar.warning("⚠️ No ML model found — using rule-based fallback")
    return None

ml_model_data = load_ml_model()

# ------------------- THINGSPEAK CONFIG -------------------
THINGSPEAK_CHANNEL_ID = "3125494"
THINGSPEAK_READ_API_KEY = ""  # optional if channel is public

def get_latest_data():
    """Get the latest ThingSpeak feed"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'temperature': float(data.get('field1', 0)),
                'humidity': float(data.get('field2', 0)),
                'soil_moisture': float(data.get('field3', 0)),
                'light_intensity': int(float(data.get('field4', 0))),
                'created_at': data.get('created_at', datetime.now().isoformat())
            }
    except Exception as e:
        st.sidebar.error(f"ThingSpeak error: {e}")
    return None

def get_historical_data(n=50):
    """Fetch recent ThingSpeak feeds"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?results={n}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            feeds = data.get("feeds", [])
            if len(feeds) == 0:
                return pd.DataFrame()
            df = pd.DataFrame(feeds)
            df = df.rename(columns={
                "field1": "temperature",
                "field2": "humidity",
                "field3": "soil_moisture",
                "field4": "light_intensity"
            })
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.dropna(subset=["temperature", "humidity", "soil_moisture", "light_intensity"])
            df = df.astype({
                "temperature": float,
                "humidity": float,
                "soil_moisture": float,
                "light_intensity": float
            })
            return df
    except Exception as e:
        st.error(f"Error fetching ThingSpeak data: {e}")
    return pd.DataFrame()

# ------------------- PREDICTION -------------------
def predict_irrigation(temperature, soil_moisture, humidity, light_intensity, crop="tomato"):
    """Predict irrigation using ML or fallback rules"""
    if ml_model_data is None:
        return rule_based_irrigation(temperature, soil_moisture, humidity, light_intensity)

    try:
        model = ml_model_data['model']
        scaler = ml_model_data['scaler']
        crop_encoder = ml_model_data.get('crop_encoder')
        pump_encoder = ml_model_data.get('pump_encoder')

        # Encode crop
        crop_mapping = {'tomato': 0, 'cucumber': 1, 'pepper': 2, 'lettuce': 3}
        crop_encoded = crop_mapping.get(crop.lower(), 0)

        # Prepare features
        X = np.array([[temperature, soil_moisture, humidity, light_intensity, crop_encoded]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        conf = np.max(probs)

        decision = pump_encoder.inverse_transform([pred])[0] if pump_encoder else ("yes" if pred == 1 else "no")

        return {
            'irrigation_decision': decision,
            'confidence': conf,
            'prob_yes': probs[1],
            'prob_no': probs[0],
            'model_used': 'RandomForest'
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return rule_based_irrigation(temperature, soil_moisture, humidity, light_intensity)

def rule_based_irrigation(temperature, soil_moisture, humidity, light_intensity):
    """Simple fallback rules"""
    if soil_moisture < 45:
        return {'irrigation_decision': 'yes', 'confidence': 0.95, 'model_used': 'RuleBased'}
    elif soil_moisture > 85:
        return {'irrigation_decision': 'no', 'confidence': 0.95, 'model_used': 'RuleBased'}
    elif soil_moisture < 55 and temperature > 30:
        return {'irrigation_decision': 'yes', 'confidence': 0.85, 'model_used': 'RuleBased'}
    elif soil_moisture < 60 and light_intensity > 700:
        return {'irrigation_decision': 'yes', 'confidence': 0.8, 'model_used': 'RuleBased'}
    else:
        return {'irrigation_decision': 'no', 'confidence': 0.75, 'model_used': 'RuleBased'}

# ------------------- UI: LIVE DATA -------------------
st.header("📡 Live ESP32 Data")
latest = get_latest_data()

if latest:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°C")
    col2.metric("💧 Humidity", f"{latest['humidity']:.1f}%")
    col3.metric("🌱 Soil Moisture", f"{latest['soil_moisture']:.1f}%")
    col4.metric("💡 Light", f"{latest['light_intensity']}")
else:
    st.warning("⚠️ No live data received yet.")

# ------------------- PREDICTION -------------------
if latest:
    st.subheader("🎯 AI Irrigation Decision")
    crop = st.sidebar.selectbox("Select crop type:", ["tomato", "cucumber", "pepper", "lettuce"], index=0)
    result = predict_irrigation(
        latest['temperature'], latest['soil_moisture'],
        latest['humidity'], latest['light_intensity'], crop
    )

    if result['irrigation_decision'] == 'yes':
        st.error("🚨 **IRRIGATION NEEDED**")
    else:
        st.success("✅ **NO IRRIGATION NEEDED**")

    st.write(f"**Confidence:** {result['confidence']:.1%}")
    st.progress(result['confidence'])
    st.caption(f"Model: {result['model_used']} | Crop: {crop.title()}")

# ------------------- HISTORICAL CHARTS -------------------
st.markdown("---")
st.header("📊 Historical Data (ThingSpeak)")

df = get_historical_data(100)
if not df.empty:
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['created_at'], y=df['temperature'], name="Temperature (°C)", line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=df['created_at'], y=df['humidity'], name="Humidity (%)", line=dict(color='blue'), yaxis="y2"))
        fig1.update_layout(
            title="Temperature & Humidity",
            xaxis_title="Time",
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(df, x="created_at", y="soil_moisture", title="Soil Moisture Over Time")
        fig2.add_hrect(y0=45, y1=85, fillcolor="green", opacity=0.1,
                       annotation_text="Optimal Range", annotation_position="top left")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.line(df, x="created_at", y="light_intensity", title="Light Intensity Over Time")
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("No historical data yet — waiting for ThingSpeak updates.")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("### 💧 Smart Irrigation AI System — Powered by ThingSpeak + ESP32")
st.caption("Real-time monitoring and ML-based irrigation decisions")
