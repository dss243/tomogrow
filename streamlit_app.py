import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import supabase
import numpy as np
import time

# Page config MUST be first
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# Initialize Supabase
@st.cache_resource
def init_supabase():
    try:
        client = supabase.create_client(
            "https://rcptkfgiiwgskbegdcih.supabase.co",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJjcHRrZmdpaXdnc2tiZWdkY2loIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIyMDQ3MDQsImV4cCI6MjA3Nzc4MDcwNH0.80h1LiXUTsF0TIzqbs7fVQvJrIZ-8XUEWuY-HeGycbs"
        )
        return client
    except Exception as e:
        st.error(f"Supabase initialization failed: {e}")
        return None

supabase_client = init_supabase()

def predict_irrigation(temperature, soil_moisture, humidity, light_intensity):
    """Simple rule-based irrigation prediction"""
    if soil_moisture < 45:
        decision = "yes"
        confidence = 0.95
    elif soil_moisture > 85:
        decision = "no" 
        confidence = 0.95
    elif soil_moisture < 55 and temperature > 30:
        decision = "yes"
        confidence = 0.85
    elif soil_moisture < 60 and light_intensity > 700:
        decision = "yes"
        confidence = 0.80
    elif soil_moisture > 75 and temperature < 20:
        decision = "no"
        confidence = 0.85
    else:
        decision = "no"
        confidence = 0.75
    
    return {
        'irrigation_prediction': decision,
        'irrigation_decision': decision,
        'confidence_level': round(confidence, 4),
        'soil_moisture_level': soil_moisture
    }

def get_latest_esp32_data():
    """Get the latest data from ESP32 device"""
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            if response.data:
                return response.data[0]
    except Exception as e:
        pass
    return None

def get_historical_data(limit=50):
    """Get historical data from Supabase"""
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        return []
    except Exception as e:
        if "does not exist" in str(e):
            return []
        return []

# Dashboard UI
st.title("🌱 Smart Irrigation Monitoring Dashboard")
st.markdown("---")

# Auto-refresh every 10 seconds
st.runtime.legacy_caching.clear_cache()

# Live Data Section
st.header("📡 Live ESP32 Data")

latest_data = get_latest_esp32_data()

if latest_data:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{latest_data.get('temperature', 0):.1f}°C")
    
    with col2:
        st.metric("💧 Humidity", f"{latest_data.get('humidity', 0):.1f}%")
    
    with col3:
        soil_moisture = latest_data.get('soil_moisture', 0)
        st.metric("🌱 Soil Moisture", f"{soil_moisture:.1f}%")
    
    with col4:
        st.metric("💡 Light", f"{latest_data.get('light_intensity', 0)}")
    
    # Make prediction
    prediction = predict_irrigation(
        latest_data.get('temperature', 0),
        soil_moisture,
        latest_data.get('humidity', 0),
        latest_data.get('light_intensity', 0)
    )
    
    st.success(f"🤖 **AI Prediction:** {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    
    # Show when data was last updated
    if 'timestamp' in latest_data:
        timestamp = pd.to_datetime(latest_data['timestamp'])
        st.caption(f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Auto-refreshes every 10 seconds)")
    
else:
    st.info("📡 Waiting for ESP32 data...")
    st.info("Make sure your ESP32 is running and sending data to Supabase")

# Rest of your dashboard...
st.markdown("---")
st.header("📊 Historical Data")

historical_data = get_historical_data()
if historical_data:
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    tab1, tab2 = st.tabs(["Soil Moisture", "All Sensors"])
    
    with tab1:
        fig = px.line(df, x='timestamp', y='soil_moisture', 
                     title='Soil Moisture Over Time',
                     labels={'soil_moisture': 'Soil Moisture (%)', 'timestamp': 'Time'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.dataframe(df[['timestamp', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'device_id']].head(10))
else:
    st.info("No historical data yet. ESP32 data will appear here automatically.")

# Manual input section
st.markdown("---")
st.header("🧪 Manual Sensor Input")

with st.form("manual_form"):
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0)
    with col2:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
        light = st.slider("Light Intensity", 0, 1000, 500)
    
    if st.form_submit_button("Test Prediction"):
        prediction = predict_irrigation(temp, moisture, humidity, light)
        st.info(f"Prediction: {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")

# Auto-refresh
time.sleep(10)
st.runtime.legacy_caching.clear_cache()
