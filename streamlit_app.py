import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import supabase
import requests
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

# Page config
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []

# Initialize Supabase
@st.cache_resource
def init_supabase():
    try:
        client = supabase.create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
        return client
    except Exception as e:
        st.error(f"Supabase initialization failed: {e}")
        return None

supabase_client = init_supabase()

# Simple ML Model for Prediction
@st.cache_resource
def load_irrigation_model():
    """Create a simple irrigation prediction model"""
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic sensor data
    temperature = np.random.normal(25, 8, n_samples)
    soil_moisture = np.random.normal(60, 20, n_samples)
    humidity = np.random.normal(65, 15, n_samples)
    light_intensity = np.random.normal(500, 200, n_samples)
    
    # Simple rules for irrigation decision
    needs_irrigation = (
        (soil_moisture < 50) | 
        ((soil_moisture < 60) & (temperature > 30)) |
        ((soil_moisture < 55) & (light_intensity > 700))
    )
    
    X = np.column_stack([temperature, soil_moisture, humidity, light_intensity])
    y = needs_irrigation.astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model

model = load_irrigation_model()

def predict_irrigation(temperature, soil_moisture, humidity, light_intensity):
    """Make irrigation prediction"""
    try:
        features = np.array([[temperature, soil_moisture, humidity, light_intensity]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        confidence = probabilities[prediction]
        decision = "yes" if prediction == 1 else "no"
        
        # Adjust confidence based on soil moisture
        if soil_moisture < 45:
            decision = "yes"
            confidence = max(confidence, 0.9)
        elif soil_moisture > 85:
            decision = "no"
            confidence = max(confidence, 0.9)
        
        return {
            'irrigation_decision': decision,
            'confidence_level': round(float(confidence), 4),
            'soil_moisture_level': soil_moisture,
            'probabilities': {
                'no': round(float(probabilities[0]), 4),
                'yes': round(float(probabilities[1]), 4)
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

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
        st.error(f"Storage error: {e}")
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
        st.error(f"Data fetch error: {e}")
        return []

# Dashboard UI
st.title("Smart Irrigation Monitoring Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
device_id = st.sidebar.text_input("Device ID", "simulated_001")

st.sidebar.header("Manual Sensor Input")
with st.sidebar.form("sensor_form"):
    st.subheader("Simulate Sensor Data")
    temp = st.slider("Temperature (C)", 0.0, 50.0, 25.0)
    moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
    light = st.slider("Light Intensity", 0, 1000, 500)
    crop = st.selectbox("Crop Type", ["tomato", "potato", "lettuce", "cucumber"])
    
    submitted = st.form_submit_button("Send Sensor Data & Predict")
    
    if submitted:
        # Create sensor data
        sensor_data = {
            'crop_type': crop,
            'temperature': temp,
            'soil_moisture': moisture,
            'humidity': humidity,
            'light_intensity': light,
            'device_id': device_id
        }
        
        # Store data
        if store_sensor_data(sensor_data):
            st.sidebar.success("Data stored successfully!")
            
            # Make prediction
            prediction = predict_irrigation(temp, moisture, humidity, light)
            if prediction:
                st.sidebar.info(f"AI Decision: {prediction['irrigation_decision'].upper()}")
                st.sidebar.info(f"Confidence: {prediction['confidence_level']:.1%}")
                st.sidebar.info(f"Soil Moisture: {moisture}%")

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)

# Get latest data
historical_data = get_historical_data(limit=1)
if historical_data:
    latest = historical_data[0]
    
    with col1:
        st.metric("Temperature", f"{latest['temperature']:.1f}C")
    
    with col2:
        st.metric("Soil Moisture", f"{latest['soil_moisture']:.1f}%")
    
    with col3:
        st.metric("Humidity", f"{latest['humidity']:.1f}%")
    
    with col4:
        st.metric("Light", f"{latest['light_intensity']}")

# Charts Section
st.markdown("---")
st.header("Historical Trends")

historical_data = get_historical_data(limit=100)
if historical_data:
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs(["Soil Moisture", "Temperature", "Humidity", "Light"])
    
    with tab1:
        fig_moisture = px.line(
            df, x='timestamp', y='soil_moisture',
            title='Soil Moisture Over Time',
            labels={'soil_moisture': 'Soil Moisture (%)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_moisture, use_container_width=True)
    
    with tab2:
        fig_temp = px.line(
            df, x='timestamp', y='temperature',
            title='Temperature Over Time',
            labels={'temperature': 'Temperature (C)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with tab3:
        fig_humidity = px.line(
            df, x='timestamp', y='humidity',
            title='Humidity Over Time',
            labels={'humidity': 'Humidity (%)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    with tab4:
        fig_light = px.line(
            df, x='timestamp', y='light_intensity',
            title='Light Intensity Over Time',
            labels={'light_intensity': 'Light Intensity', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_light, use_container_width=True)

else:
    st.info("No historical data yet. Use the sidebar to simulate sensor data!")

# Prediction History
st.markdown("---")
st.header("AI Prediction Log")

# Simulate some prediction history
if st.button("Generate Sample Predictions"):
    sample_predictions = []
    for i in range(10):
        sample_data = {
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
            'temperature': np.random.uniform(20, 35),
            'soil_moisture': np.random.uniform(40, 80),
            'irrigation_decision': np.random.choice(['yes', 'no']),
            'confidence_level': np.random.uniform(0.7, 0.95)
        }
        sample_predictions.append(sample_data)
    
    df_pred = pd.DataFrame(sample_predictions)
    st.dataframe(df_pred)

# Real-time Data Simulation
st.markdown("---")
st.header("Real-time Simulation")

if st.button("Start Live Simulation"):
    placeholder = st.empty()
    
    for i in range(20):
        # Generate random sensor data
        sim_data = {
            'crop_type': 'tomato',
            'temperature': np.random.normal(25, 5),
            'soil_moisture': np.random.normal(60, 15),
            'humidity': np.random.normal(65, 10),
            'light_intensity': np.random.normal(500, 100),
            'device_id': 'simulated'
        }
        
        # Store and predict
        store_sensor_data(sim_data)
        prediction = predict_irrigation(
            sim_data['temperature'], 
            sim_data['soil_moisture'],
            sim_data['humidity'],
            sim_data['light_intensity']
        )
        
        with placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write("Latest Sensor Data:")
                st.json(sim_data)
            with col2:
                st.write("AI Prediction:")
                st.json(prediction)
        
        st.runtime.legacy_caching.clear_cache()

# Footer
st.markdown("---")
st.markdown("Smart Irrigation System | AI-Powered Decisions")
