import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import supabase
import numpy as np
import paho.mqtt.client as mqtt
import json
import threading

# Page config MUST be first
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "tomogrow/sensor/data"

# Global variable to store latest sensor data
latest_sensor_data = None

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

# MQTT Functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
    else:
        st.error(f"MQTT Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global latest_sensor_data
    try:
        data = json.loads(msg.payload.decode())
        latest_sensor_data = data
        st.runtime.legacy_caching.clear_cache()  # Refresh the app
    except Exception as e:
        st.error(f"MQTT message error: {e}")

def start_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        st.error(f"MQTT connection failed: {e}")

# Start MQTT in a separate thread
if 'mqtt_started' not in st.session_state:
    st.session_state.mqtt_started = True
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()

# Your existing functions
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

def store_sensor_data(data):
    """Store sensor data in Supabase"""
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data").insert({
                "crop_type": data['crop_type'],
                "temperature": data['temperature'],
                "soil_moisture": data['soil_moisture'],
                "humidity": data['humidity'],
                "light_intensity": data['light_intensity'],
                "device_id": data.get('device_id', 'simulated'),
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            return True
    except Exception as e:
        return False
    return False

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
st.title("Smart Irrigation Monitoring Dashboard")
st.markdown("---")

# MQTT Live Data Section
st.header("📡 MQTT Live Data from ESP32")

if latest_sensor_data:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🌡️ Temperature", f"{latest_sensor_data.get('temperature', 0):.1f}°C")
    
    with col2:
        st.metric("💧 Humidity", f"{latest_sensor_data.get('humidity', 0):.1f}%")
    
    with col3:
        st.metric("🌱 Soil Moisture", f"{latest_sensor_data.get('soil_moisture', 0):.1f}%")
    
    with col4:
        st.metric("💡 Light", f"{latest_sensor_data.get('light_intensity', 0)}")
    
    with col5:
        st.metric("🌫️ Air Quality", f"{latest_sensor_data.get('air_quality', 0):.1f}")
    
    # Make prediction with MQTT data
    prediction = predict_irrigation(
        latest_sensor_data.get('temperature', 0),
        latest_sensor_data.get('soil_moisture', 0),
        latest_sensor_data.get('humidity', 0),
        latest_sensor_data.get('light_intensity', 0)
    )
    
    st.success(f"🤖 **Live Prediction:** {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    
    # Store the MQTT data in Supabase
    if store_sensor_data({
        'crop_type': latest_sensor_data.get('crop_type', 'tomato'),
        'temperature': latest_sensor_data.get('temperature', 0),
        'soil_moisture': latest_sensor_data.get('soil_moisture', 0),
        'humidity': latest_sensor_data.get('humidity', 0),
        'light_intensity': latest_sensor_data.get('light_intensity', 0),
        'device_id': latest_sensor_data.get('device_id', 'esp32_mqtt')
    }):
        st.info("✅ MQTT data stored in database")
    
else:
    st.info("📡 Waiting for MQTT data from ESP32...")
    st.info("Make sure your ESP32 is running and connected to WiFi")

# Rest of your existing dashboard code continues here...
# [Include all your existing sidebar, charts, etc.]
