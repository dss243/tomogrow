import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
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
            # First, let's check what columns exist in the table
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                # Get the actual column names from the first record
                available_columns = list(response.data[0].keys())
                st.sidebar.info(f"Available columns: {', '.join(available_columns)}")
                
                # Now get the latest data without ordering by timestamp
                latest_response = supabase_client.table("sensor_data")\
                    .select("*")\
                    .eq("device_id", "ESP32_TOMOGROW_001")\
                    .limit(1)\
                    .execute()
                
                if latest_response.data and len(latest_response.data) > 0:
                    return latest_response.data[0]
                    
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
    return None

def get_historical_data(limit=50):
    """Get historical data from Supabase"""
    try:
        if supabase_client:
            # Get data without ordering by timestamp
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .limit(limit)\
                .execute()
            return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
    return []

# Dashboard UI
st.title("🌱 Smart Irrigation Monitoring Dashboard")
st.markdown("---")

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000, key="data_refresh")
    st.success("🔄 Auto-refresh enabled (10 seconds)")
except:
    st.info("🔄 Auto-refresh not available. Refresh page manually for updates.")

# Debug info in sidebar
st.sidebar.header("🔧 Debug Info")
st.sidebar.write("Check your database schema and fix column names if needed")

# Live Data Section
st.header("📡 Live ESP32 Data")

latest_data = get_latest_esp32_data()

if latest_data:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature = latest_data.get('temperature', 0)
        st.metric("🌡️ Temperature", f"{temperature:.1f}°C")
    
    with col2:
        humidity = latest_data.get('humidity', 0)
        st.metric("💧 Humidity", f"{humidity:.1f}%")
    
    with col3:
        soil_moisture = latest_data.get('soil_moisture', 0)
        st.metric("🌱 Soil Moisture", f"{soil_moisture:.1f}%")
    
    with col4:
        light_intensity = latest_data.get('light_intensity', 0)
        st.metric("💡 Light", f"{light_intensity}")
    
    # Make prediction
    prediction = predict_irrigation(
        temperature,
        soil_moisture,
        humidity,
        light_intensity
    )
    
    # Show prediction with color coding
    if prediction['irrigation_decision'] == 'yes':
        st.error(f"🚨 **IRRIGATION NEEDED:** {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    else:
        st.success(f"✅ **NO IRRIGATION NEEDED:** {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    
    # Show data details
    with st.expander("View Raw Data"):
        st.json(latest_data)
    
else:
    st.warning("📡 Waiting for ESP32 data...")
    st.info("If data is being sent but not showing, check your database column names")
    
    # Show sample data for testing
    with st.expander("Test with sample data"):
        st.write("If ESP32 is not connected, you can still test the system:")
        sample_temp = st.slider("Sample Temperature", 0.0, 50.0, 25.0)
        sample_moisture = st.slider("Sample Soil Moisture", 0.0, 100.0, 60.0)
        
        sample_prediction = predict_irrigation(sample_temp, sample_moisture, 65, 500)
        if sample_prediction['irrigation_decision'] == 'yes':
            st.error(f"Sample Prediction: {sample_prediction['irrigation_decision'].upper()} (Confidence: {sample_prediction['confidence_level']:.1%})")
        else:
            st.success(f"Sample Prediction: {sample_prediction['irrigation_decision'].upper()} (Confidence: {sample_prediction['confidence_level']:.1%})")

# Historical Data Section
st.markdown("---")
st.header("📊 Historical Data")

historical_data = get_historical_data()
if historical_data and len(historical_data) > 0:
    df = pd.DataFrame(historical_data)
    
    # Check if we have a timestamp column or use id for ordering
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'])
    elif 'id' in df.columns:
        # Use id as proxy for time order
        df = df.sort_values('id', ascending=False)
        df['timestamp'] = range(len(df))  # Create dummy timestamp
    else:
        df['timestamp'] = range(len(df))  # Create dummy timestamp
    
    st.success(f"✅ Found {len(df)} historical records")
    
    tab1, tab2, tab3 = st.tabs(["Soil Moisture", "Temperature & Humidity", "Data Table"])
    
    with tab1:
        if 'soil_moisture' in df.columns:
            fig_moisture = px.line(df, x='timestamp', y='soil_moisture', 
                                  title='Soil Moisture Over Time',
                                  labels={'soil_moisture': 'Soil Moisture (%)', 'timestamp': 'Record Index'})
            # Add irrigation zones
            fig_moisture.add_hrect(y0=0, y1=45, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Irrigation Needed")
            fig_moisture.add_hrect(y0=45, y1=85, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Optimal Range")
            fig_moisture.add_hrect(y0=85, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Too Wet")
            st.plotly_chart(fig_moisture, use_container_width=True)
        else:
            st.warning("No soil_moisture column found in data")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'temperature' in df.columns:
                fig_temp = px.line(df, x='timestamp', y='temperature', 
                                  title='Temperature Over Time',
                                  labels={'temperature': 'Temperature (°C)', 'timestamp': 'Record Index'})
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("No temperature column found")
        
        with col2:
            if 'humidity' in df.columns:
                fig_humidity = px.line(df, x='timestamp', y='humidity', 
                                      title='Humidity Over Time',
                                      labels={'humidity': 'Humidity (%)', 'timestamp': 'Record Index'})
                st.plotly_chart(fig_humidity, use_container_width=True)
            else:
                st.warning("No humidity column found")
    
    with tab3:
        # Display available data
        display_columns = ['device_id', 'temperature', 'humidity', 'soil_moisture', 'light_intensity']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            st.dataframe(df[available_columns].head(20))
        else:
            st.warning("No expected columns found in data")
            st.write("Available columns:", df.columns.tolist())
        
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
        st.error(f"Dry Soil: {prediction['irrigation_decision'].upper()} (Confidence: {prediction['confidence_level']:.1%})")
    
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
            if custom_prediction['irrigation_decision'] == 'yes':
                st.error(f"Custom Prediction: {custom_prediction['irrigation_decision'].upper()} (Confidence: {custom_prediction['confidence_level']:.1%})")
            else:
                st.success(f"Custom Prediction: {custom_prediction['irrigation_decision'].upper()} (Confidence: {custom_prediction['confidence_level']:.1%})")

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

# Database Setup Help
st.markdown("---")
st.header("🗄️ Database Setup")

with st.expander("Click here if you need to set up your Supabase table"):
    st.markdown("""
    **Required Table Schema:**
    ```sql
    CREATE TABLE sensor_data (
      id BIGSERIAL PRIMARY KEY,
      device_id TEXT,
      temperature DECIMAL,
      humidity DECIMAL,
      soil_moisture DECIMAL,
      light_intensity INTEGER,
      created_at TIMESTAMP DEFAULT NOW()
    );
    ```
    
    **Or use these alternative column names:**
    - Use `created_at` instead of `timestamp`
    - Or let the system auto-detect your columns
    """)

# Footer
st.markdown("---")
st.markdown("### 💧 Smart Irrigation System | 🤖 AI-Powered Decisions")
st.markdown("*Real-time monitoring and intelligent irrigation predictions*")
