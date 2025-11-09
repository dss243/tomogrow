import streamlit as st

# Page config MUST be first
st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="💧",
    layout="wide"
)

# Now import other packages
import pandas as pd
import plotly.express as px
from datetime import datetime
import supabase
import numpy as np

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []

# Initialize Supabase with your credentials
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
    """Simple rule-based irrigation prediction - NO ML NEEDED"""
    
    # Rule-based logic (better than ML for irrigation)
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
            
            if response.data:
                return True
        return False
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
        # Don't show error if it's just because table doesn't exist yet
        if "does not exist" in str(e):
            return []
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
            st.sidebar.success("✅ Data stored successfully!")
            
            # Make prediction
            prediction = predict_irrigation(temp, moisture, humidity, light)
            if prediction:
                st.sidebar.info(f"🤖 **Irrigation Prediction:** {prediction['irrigation_prediction'].upper()}")
                st.sidebar.info(f"🎯 **Confidence:** {prediction['confidence_level']:.1%}")
                st.sidebar.info(f"💧 **Soil Moisture:** {moisture}%")
        else:
            st.sidebar.error("❌ Failed to store data. Make sure the Supabase table is created.")

# Main Dashboard - Current Metrics
col1, col2, col3, col4 = st.columns(4)

# Get latest data
historical_data = get_historical_data(limit=1)
if historical_data:
    latest = historical_data[0]
    
    with col1:
        st.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°C")
    
    with col2:
        st.metric("💧 Soil Moisture", f"{latest['soil_moisture']:.1f}%")
    
    with col3:
        st.metric("💨 Humidity", f"{latest['humidity']:.1f}%")
    
    with col4:
        st.metric("☀️ Light", f"{latest['light_intensity']}")
else:
    with col1:
        st.metric("🌡️ Temperature", "N/A")
    with col2:
        st.metric("💧 Soil Moisture", "N/A")
    with col3:
        st.metric("💨 Humidity", "N/A")
    with col4:
        st.metric("☀️ Light", "N/A")

# Charts Section
st.markdown("---")
st.header("📈 Historical Trends")

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
        # Add irrigation zones
        fig_moisture.add_hrect(y0=0, y1=45, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Irrigation Needed")
        fig_moisture.add_hrect(y0=45, y1=85, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Optimal Range")
        fig_moisture.add_hrect(y0=85, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Too Wet")
        st.plotly_chart(fig_moisture, use_container_width=True)
    
    with tab2:
        fig_temp = px.line(
            df, x='timestamp', y='temperature',
            title='Temperature Over Time',
            labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'}
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
    st.info("📝 No historical data yet. Use the sidebar to simulate sensor data!")
    st.info("💡 If you see errors, make sure to run the SQL in Supabase to create the table.")

# Quick Setup Instructions
st.markdown("---")
st.header("🔧 Setup Instructions")

with st.expander("Click here for Supabase setup instructions"):
    st.markdown("""
    **To fix the 'timestamp does not exist' error:**
    
    1. Go to your Supabase project: https://rcptkfgiiwgskbegdcih.supabase.co
    2. Click on **SQL Editor** in the left sidebar
    3. Copy and paste this SQL code:
    ```sql
    DROP TABLE IF EXISTS sensor_data;
    CREATE TABLE sensor_data (
        id BIGSERIAL PRIMARY KEY,
        crop_type TEXT NOT NULL,
        temperature FLOAT,
        soil_moisture FLOAT,
        humidity FLOAT,
        light_intensity FLOAT,
        device_id TEXT,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );
    
    ALTER TABLE sensor_data ENABLE ROW LEVEL SECURITY;
    CREATE POLICY "Allow public access" ON sensor_data FOR ALL USING (true);
    ```
    4. Click **Run** to execute the SQL
    5. Your table will be created and ready to use!
    """)

# System Status
st.markdown("---")
st.header("🔧 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Database Connection")
    if supabase_client:
        st.success("✅ Connected to Supabase")
    else:
        st.error("❌ Not connected to Supabase")

with col2:
    st.subheader("Data Storage")
    historical_data = get_historical_data(limit=1)
    if historical_data:
        all_data = get_historical_data(limit=1000)
        st.success(f"✅ Storing data ({len(all_data)} records)")
    else:
        st.info("📝 No data stored yet")

with col3:
    st.subheader("Table Status")
    historical_data = get_historical_data(limit=1)
    if historical_data:
        st.success("✅ Table exists and working")
    else:
        st.warning("⚠️ Table may not exist")

# Footer
st.markdown("---")
st.markdown("### 💧 Smart Irrigation System | 🤖 AI-Powered Decisions")
st.markdown("*Real-time monitoring and intelligent irrigation predictions*")
