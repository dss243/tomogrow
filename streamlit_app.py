import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import supabase
import numpy as np

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

# Auto-refresh using streamlit_autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    # Run autorefresh every 10 seconds (10000 milliseconds)
    st_autorefresh(interval=10000, key="data_refresh")
except:
    st.info("🔄 Auto-refresh not available. Refresh page manually for updates.")

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
        st.caption(f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
else:
    st.info("📡 Waiting for ESP32 data...")
    st.info("Make sure your ESP32 is running and sending data to Supabase")
    
    # Show sample data for testing
    with st.expander("Test with sample data"):
        st.write("If ESP32 is not connected, you can still test the system:")
        sample_temp = st.slider("Sample Temperature", 0.0, 50.0, 25.0)
        sample_moisture = st.slider("Sample Soil Moisture", 0.0, 100.0, 60.0)
        
        sample_prediction = predict_irrigation(sample_temp, sample_moisture, 65, 500)
        st.info(f"Sample Prediction: {sample_prediction['irrigation_decision'].upper()} (Confidence: {sample_prediction['confidence_level']:.1%})")

# Historical Data Section
st.markdown("---")
st.header("📊 Historical Data")

historical_data = get_historical_data()
if historical_data:
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    tab1, tab2, tab3 = st.tabs(["Soil Moisture", "Temperature & Humidity", "Data Table"])
    
    with tab1:
        fig_moisture = px.line(df, x='timestamp', y='soil_moisture', 
                              title='Soil Moisture Over Time',
                              labels={'soil_moisture': 'Soil Moisture (%)', 'timestamp': 'Time'})
        # Add irrigation zones
        fig_moisture.add_hrect(y0=0, y1=45, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Irrigation Needed")
        fig_moisture.add_hrect(y0=45, y1=85, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Optimal Range")
        fig_moisture.add_hrect(y0=85, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Too Wet")
        st.plotly_chart(fig_moisture, use_container_width=True)
    
    with tab2:
        fig_temp = px.line(df, x='timestamp', y='temperature', 
                          title='Temperature Over Time',
                          labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'})
        st.plotly_chart(fig_temp, use_container_width=True)
        
        fig_humidity = px.line(df, x='timestamp', y='humidity', 
                              title='Humidity Over Time',
                              labels={'humidity': 'Humidity (%)', 'timestamp': 'Time'})
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    with tab3:
        st.dataframe(df[['timestamp', 'device_id', 'temperature', 'humidity', 'soil_moisture', 'light_intensity']].head(20))
        
else:
    st.info("No historical data yet. ESP32 data will appear here automatically.")

# Manual Testing Section
st.markdown("---")
st.header("🧪 Manual Testing")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Quick Test")
    if st.button("Test Dry Soil (Needs Irrigation)"):
        prediction = predict_irrigation(30, 35, 60, 500)
        st.success(f"Dry Soil: {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    
    if st.button("Test Wet Soil (No Irrigation)"):
        prediction = predict_irrigation(25, 90, 60, 500)
        st.success(f"Wet Soil: {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")

with col2:
    st.subheader("Custom Test")
    with st.form("custom_test"):
        custom_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, key="custom_temp")
        custom_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0, key="custom_moisture")
        
        if st.form_submit_button("Get Prediction"):
            custom_prediction = predict_irrigation(custom_temp, custom_moisture, 65, 500)
            st.info(f"Custom Prediction: {custom_prediction['irrigation_decision'].upper()} (Confidence: {custom_prediction['confidence_level']:.1%})")

# System Status
st.markdown("---")
st.header("🔧 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.subheader("Database Connection")
    if supabase_client:
        st.success("✅ Connected to Supabase")
    else:
        st.error("❌ Not connected to Supabase")

with status_col2:
    st.subheader("ESP32 Status")
    if latest_data:
        st.success("✅ Receiving data from ESP32")
    else:
        st.warning("⚠️ Waiting for ESP32 data")

with status_col3:
    st.subheader("AI System")
    st.success("✅ Rule-based AI Active")

# Footer
st.markdown("---")
st.markdown("### 💧 Smart Irrigation System | 🤖 AI-Powered Decisions")
st.markdown("*Real-time monitoring and intelligent irrigation predictions*")
