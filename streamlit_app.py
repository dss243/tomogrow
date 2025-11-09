import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import supabase
import os

# Page config
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# Supabase configuration
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
FASTAPI_URL = st.secrets["FASTAPI_URL"]

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

def get_historical_data(device_id="ESP32_IRRIGATION_001", limit=100):
    """Fetch historical sensor data from Supabase"""
    try:
        response = supabase_client.table("sensor_data")\
            .select("*")\
            .eq("device_id", device_id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

def get_predictions(device_id="ESP32_IRRIGATION_001", limit=50):
    """Fetch prediction history"""
    try:
        response = supabase_client.table("predictions")\
            .select("*")\
            .eq("device_id", device_id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []

def make_prediction(sensor_data):
    """Make irrigation prediction using FastAPI"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/predict-irrigation",
            json=sensor_data
        )
        return response.json()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Dashboard UI
st.title("🌱 Smart Irrigation Monitoring Dashboard")

# Sidebar
st.sidebar.header("Configuration")
device_id = st.sidebar.text_input("Device ID", "ESP32_IRRIGATION_001")
refresh_rate = st.sidebar.selectbox("Refresh Rate", [30, 60, 300], index=0)

# Manual prediction section
st.sidebar.header("Manual Prediction")
with st.sidebar.form("prediction_form"):
    st.subheader("Test Irrigation Prediction")
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
    light = st.slider("Light Intensity", 0, 1000, 500)
    crop = st.selectbox("Crop Type", ["tomato", "potato", "lettuce"])
    
    if st.form_submit_button("Predict Irrigation"):
        prediction_data = {
            "crop_type": crop,
            "temperature": temp,
            "soil_moisture": moisture,
            "humidity": humidity,
            "light_intensity": light
        }
        
        result = make_prediction(prediction_data)
        if result:
            st.success(f"Decision: {result['irrigation_decision'].upper()}")
            st.info(f"Confidence: {result['confidence_level']:.2%}")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Fetch current data
data = get_historical_data(device_id, limit=1)
if data:
    current = data[0]
    
    with col1:
        st.metric(
            "Temperature", 
            f"{current['temperature']:.1f}°C",
            delta=None
        )
    
    with col2:
        st.metric(
            "Soil Moisture", 
            f"{current['soil_moisture']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Humidity", 
            f"{current['humidity']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "Light Intensity", 
            f"{current['light_intensity']}",
            delta=None
        )

# Charts
st.subheader("Historical Data Trends")
historical_data = get_historical_data(device_id, limit=50)

if historical_data:
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Soil Moisture", "Temperature & Humidity", "Light Intensity"])
    
    with tab1:
        fig_moisture = px.line(
            df, x='timestamp', y='soil_moisture',
            title='Soil Moisture Over Time',
            labels={'soil_moisture': 'Soil Moisture (%)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_moisture, use_container_width=True)
    
    with tab2:
        fig_temp_hum = go.Figure()
        fig_temp_hum.add_trace(go.Scatter(
            x=df['timestamp'], y=df['temperature'],
            name='Temperature', line=dict(color='red')
        ))
        fig_temp_hum.add_trace(go.Scatter(
            x=df['timestamp'], y=df['humidity'],
            name='Humidity', line=dict(color='blue'),
            yaxis='y2'
        ))
        fig_temp_hum.update_layout(
            title='Temperature & Humidity',
            yaxis=dict(title='Temperature (°C)'),
            yaxis2=dict(title='Humidity (%)', overlaying='y', side='right')
        )
        st.plotly_chart(fig_temp_hum, use_container_width=True)
    
    with tab3:
        fig_light = px.line(
            df, x='timestamp', y='light_intensity',
            title='Light Intensity Over Time',
            labels={'light_intensity': 'Light Intensity', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_light, use_container_width=True)

# Prediction history
st.subheader("Irrigation Prediction History")
predictions = get_predictions(device_id)
if predictions:
    pred_df = pd.DataFrame(predictions)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    
    # Display recent predictions
    recent_preds = pred_df.head(10)[['timestamp', 'irrigation_decision', 'confidence_level', 'soil_moisture_level']]
    st.dataframe(recent_preds.style.format({
        'confidence_level': '{:.2%}',
        'soil_moisture_level': '{:.1f}%'
    }))

# Auto-refresh
st.sidebar.info(f"Auto-refreshing every {refresh_rate} seconds")
st.runtime.legacy_caching.clear_cache()