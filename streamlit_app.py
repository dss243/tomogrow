import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    """Advanced rule-based irrigation prediction"""
    # Calculate multiple factors
    moisture_factor = max(0, (45 - soil_moisture) / 45)  # 0-1, higher when dry
    temp_factor = max(0, (temperature - 25) / 15)        # Higher when hot
    light_factor = min(1, light_intensity / 1000)        # Higher when bright
    humidity_factor = max(0, (40 - humidity) / 40)       # Higher when dry air
    
    # Weighted decision score
    score = (moisture_factor * 0.5 + 
             temp_factor * 0.2 + 
             light_factor * 0.2 + 
             humidity_factor * 0.1)
    
    # Dynamic confidence based on agreement between factors
    factors = [moisture_factor, temp_factor, light_factor, humidity_factor]
    confidence_variance = np.std(factors)
    base_confidence = 0.7 + (0.25 * (1 - confidence_variance))
    
    # Make decision
    if score > 0.4:
        decision = "yes"
        confidence = min(0.95, base_confidence + score * 0.3)
    else:
        decision = "no"
        confidence = min(0.95, base_confidence + (1 - score) * 0.3)
    
    # Special overrides
    if soil_moisture < 30:
        decision = "yes"
        confidence = 0.98
    elif soil_moisture > 90:
        decision = "no"
        confidence = 0.98
    
    return {
        'irrigation_decision': decision,
        'confidence_level': round(confidence, 4),
        'score': round(score, 3),
        'factors': {
            'moisture_factor': round(moisture_factor, 3),
            'temp_factor': round(temp_factor, 3),
            'light_factor': round(light_factor, 3),
            'humidity_factor': round(humidity_factor, 3)
        }
    }

def get_latest_esp32_data():
    """Get the latest data from ESP32 device"""
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .order("id", desc=True)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
    except Exception as e:
        st.error(f"Error fetching latest data: {e}")
    return None

def get_historical_data(limit=100):
    """Get historical data from Supabase"""
    try:
        if supabase_client:
            response = supabase_client.table("sensor_data")\
                .select("*")\
                .eq("device_id", "ESP32_TOMOGROW_001")\
                .order("id", desc=True)\
                .limit(limit)\
                .execute()
            return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
    return []

def create_timestamp_column(df):
    """Create proper timestamp for charts"""
    # Try different possible timestamp columns
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # Create synthetic timestamp based on record order
        df = df.sort_values('id').reset_index(drop=True)
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=len(df)),
            periods=len(df),
            freq='30S'
        )
    return df

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

# Live Data Section
st.header("📡 Live ESP32 Data")

latest_data = get_latest_esp32_data()

if latest_data:
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature = latest_data.get('temperature', 0)
        if temperature is not None:
            st.metric("🌡️ Temperature", f"{float(temperature):.1f}°C", 
                     delta=f"{(float(temperature) - 25):+.1f}°C")
        else:
            st.metric("🌡️ Temperature", "N/A")
    
    with col2:
        humidity = latest_data.get('humidity', 0)
        if humidity is not None:
            st.metric("💧 Humidity", f"{float(humidity):.1f}%", 
                     delta=f"{(float(humidity) - 50):+.1f}%")
        else:
            st.metric("💧 Humidity", "N/A")
    
    with col3:
        soil_moisture = latest_data.get('soil_moisture', 0)
        if soil_moisture is not None:
            soil_moisture_float = float(soil_moisture)
            moisture_status = "🟢 Optimal" if 45 <= soil_moisture_float <= 85 else "🟡 Watch" if soil_moisture_float > 85 else "🔴 Dry"
            st.metric("🌱 Soil Moisture", f"{soil_moisture_float:.1f}%", moisture_status)
        else:
            st.metric("🌱 Soil Moisture", "N/A")
    
    with col4:
        light_intensity = latest_data.get('light_intensity', 0)
        if light_intensity is not None:
            light_intensity_int = int(light_intensity)
            light_status = "☀️ Bright" if light_intensity_int > 700 else "⛅ Moderate" if light_intensity_int > 300 else "🌙 Dark"
            st.metric("💡 Light", f"{light_intensity_int}", light_status)
        else:
            st.metric("💡 Light", "N/A")
    
    # AI Prediction
    if all(key in latest_data and latest_data[key] is not None for key in ['temperature', 'soil_moisture', 'humidity', 'light_intensity']):
        prediction = predict_irrigation(
            float(latest_data['temperature']),
            float(latest_data['soil_moisture']),
            float(latest_data['humidity']),
            int(latest_data['light_intensity'])
        )
        
        # Enhanced prediction display
        pred_col1, pred_col2 = st.columns([2, 1])
        
        with pred_col1:
            if prediction['irrigation_decision'] == 'yes':
                st.error(f"🚨 **IRRIGATION NEEDED** (Confidence: {prediction['confidence_level']:.1%})")
                st.progress(float(prediction['score']))
                st.caption(f"Decision Score: {prediction['score']:.3f}")
            else:
                st.success(f"✅ **NO IRRIGATION NEEDED** (Confidence: {prediction['confidence_level']:.1%})")
                st.progress(float(1 - prediction['score']))
                st.caption(f"Decision Score: {prediction['score']:.3f}")
        
        with pred_col2:
            with st.expander("AI Factors"):
                st.write(f"💧 Moisture: {prediction['factors']['moisture_factor']:.3f}")
                st.write(f"🌡️ Temperature: {prediction['factors']['temp_factor']:.3f}")
                st.write(f"💡 Light: {prediction['factors']['light_factor']:.3f}")
                st.write(f"💨 Humidity: {prediction['factors']['humidity_factor']:.3f}")
    else:
        st.warning("⚠️ Incomplete data for AI prediction")

else:
    st.warning("📡 Waiting for ESP32 data...")
    
    # Show sample data for testing
    with st.expander("Test with sample data"):
        col1, col2 = st.columns(2)
        with col1:
            sample_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
            sample_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0)
        with col2:
            sample_humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
            sample_light = st.slider("Light Intensity", 0, 1000, 500)
        
        sample_prediction = predict_irrigation(sample_temp, sample_moisture, sample_humidity, sample_light)
        if sample_prediction['irrigation_decision'] == 'yes':
            st.error(f"Sample Prediction: {sample_prediction['irrigation_decision'].upper()} (Confidence: {sample_prediction['confidence_level']:.1%})")
        else:
            st.success(f"Sample Prediction: {sample_prediction['irrigation_decision'].upper()} (Confidence: {sample_prediction['confidence_level']:.1%})")

# Historical Data Section
st.markdown("---")
st.header("📊 Historical Data & Analytics")

historical_data = get_historical_data(100)  # Get last 100 records

if historical_data and len(historical_data) > 0:
    df = pd.DataFrame(historical_data)
    df = create_timestamp_column(df)
    
    # Convert numeric columns safely
    numeric_columns = ['temperature', 'humidity', 'soil_moisture', 'light_intensity']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    st.success(f"✅ Analyzing {len(df)} historical records")
    
    # Calculate statistics
    avg_temp = df['temperature'].mean() if 'temperature' in df.columns and not df['temperature'].isna().all() else 0
    avg_moisture = df['soil_moisture'].mean() if 'soil_moisture' in df.columns and not df['soil_moisture'].isna().all() else 0
    avg_humidity = df['humidity'].mean() if 'humidity' in df.columns and not df['humidity'].isna().all() else 0
    
    # Statistics row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    with stat_col2:
        st.metric("Avg Soil Moisture", f"{avg_moisture:.1f}%")
    with stat_col3:
        st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
    with stat_col4:
        st.metric("Data Points", len(df))
    
    # Interactive Charts
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Soil Analysis", "🌡️ Environment", "📈 Trends", "📋 Raw Data"])
    
    with tab1:
        if 'soil_moisture' in df.columns and not df['soil_moisture'].isna().all():
            fig_soil = go.Figure()
            fig_soil.add_trace(go.Scatter(x=df['timestamp'], y=df['soil_moisture'], 
                                         mode='lines+markers', name='Soil Moisture',
                                         line=dict(color='#2E8B57', width=3)))
            
            # Add irrigation zones
            fig_soil.add_hrect(y0=0, y1=45, fillcolor="red", opacity=0.2, 
                              line_width=0, annotation_text="Irrigation Zone")
            fig_soil.add_hrect(y0=45, y1=85, fillcolor="green", opacity=0.2,
                              line_width=0, annotation_text="Optimal Zone")
            fig_soil.add_hrect(y0=85, y1=100, fillcolor="yellow", opacity=0.2,
                              line_width=0, annotation_text="Saturated Zone")
            
            fig_soil.update_layout(title='Soil Moisture Over Time with Irrigation Zones',
                                 xaxis_title='Time', yaxis_title='Soil Moisture (%)',
                                 height=400)
            st.plotly_chart(fig_soil, use_container_width=True)
            
            # Soil moisture distribution
            col1, col2 = st.columns(2)
            with col1:
                fig_dist = px.histogram(df, x='soil_moisture', 
                                      title='Soil Moisture Distribution',
                                      nbins=20, color_discrete_sequence=['#2E8B57'])
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                current_moisture = df['soil_moisture'].iloc[-1] if not df['soil_moisture'].isna().iloc[-1] else 0
                moisture_level = "🔴 DRY" if current_moisture < 45 else "🟢 OPTIMAL" if current_moisture <= 85 else "🟡 WET"
                st.metric("Current Soil Status", moisture_level, 
                         f"{current_moisture:.1f}%", delta_color="off")
        else:
            st.warning("No soil moisture data available for analysis")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'temperature' in df.columns and not df['temperature'].isna().all():
                fig_temp = px.line(df, x='timestamp', y='temperature',
                                 title='Temperature Trend',
                                 line_shape='spline')
                fig_temp.update_traces(line=dict(color='#FF6B6B', width=3))
                fig_temp.update_layout(height=300)
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("No temperature data available")
        
        with col2:
            if 'humidity' in df.columns and not df['humidity'].isna().all():
                fig_hum = px.line(df, x='timestamp', y='humidity',
                                title='Humidity Trend',
                                line_shape='spline')
                fig_hum.update_traces(line=dict(color='#4ECDC4', width=3))
                fig_hum.update_layout(height=300)
                st.plotly_chart(fig_hum, use_container_width=True)
            else:
                st.warning("No humidity data available")
    
    with tab3:
        # Multi-line trend chart
        fig_trend = go.Figure()
        
        traces_added = False
        if 'temperature' in df.columns and not df['temperature'].isna().all():
            fig_trend.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'],
                                         name='Temperature', yaxis='y1',
                                         line=dict(color='#FF6B6B')))
            traces_added = True
        
        if 'soil_moisture' in df.columns and not df['soil_moisture'].isna().all():
            fig_trend.add_trace(go.Scatter(x=df['timestamp'], y=df['soil_moisture'],
                                         name='Soil Moisture', yaxis='y2',
                                         line=dict(color='#2E8B57')))
            traces_added = True
        
        if 'humidity' in df.columns and not df['humidity'].isna().all():
            fig_trend.add_trace(go.Scatter(x=df['timestamp'], y=df['humidity'],
                                         name='Humidity', yaxis='y3',
                                         line=dict(color='#4ECDC4')))
            traces_added = True
        
        if traces_added:
            fig_trend.update_layout(
                title='Multi-Sensor Trends Over Time',
                xaxis=dict(domain=[0.1, 0.9]),
                yaxis=dict(title='Temperature (°C)', side='left', position=0.05),
                yaxis2=dict(title='Soil Moisture (%)', side='right', overlaying='y', position=0.95),
                yaxis3=dict(title='Humidity (%)', side='right', overlaying='y', position=1),
                height=500
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No sensor data available for trend analysis")
    
    with tab4:
        # Display raw data with sorting options
        sort_by = st.selectbox("Sort by:", ['id', 'temperature', 'soil_moisture'], index=0)
        ascending = st.checkbox("Ascending order", value=False)
        
        df_display = df.sort_values(sort_by, ascending=ascending)
        display_columns = [col for col in ['id', 'timestamp', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'device_id'] 
                          if col in df_display.columns]
        
        st.dataframe(df_display[display_columns].head(20), use_container_width=True)
        
else:
    st.info("No historical data yet. ESP32 data will appear here automatically.")

# Manual Testing Section
st.markdown("---")
st.header("🧪 Manual Testing & Simulation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Quick Scenarios")
    
    scenario = st.selectbox("Choose scenario:", 
                           ["Normal Day", "Hot & Dry", "Cool & Wet", "Extreme Dry", "Optimal Conditions"])
    
    if st.button("Run Scenario Analysis"):
        scenarios = {
            "Normal Day": (25, 60, 65, 500),
            "Hot & Dry": (35, 30, 40, 800),
            "Cool & Wet": (18, 85, 75, 300),
            "Extreme Dry": (30, 15, 35, 900),
            "Optimal Conditions": (22, 65, 60, 600)
        }
        
        temp, moisture, hum, light = scenarios[scenario]
        prediction = predict_irrigation(temp, moisture, hum, light)
        
        st.write(f"**Scenario:** {scenario}")
        st.write(f"**Conditions:** {temp}°C, {moisture}% soil, {hum}% humidity, {light} light")
        
        if prediction['irrigation_decision'] == 'yes':
            st.error(f"🚨 IRRIGATION NEEDED (Confidence: {prediction['confidence_level']:.1%})")
        else:
            st.success(f"✅ NO IRRIGATION NEEDED (Confidence: {prediction['confidence_level']:.1%})")

with col2:
    st.subheader("Custom Test")
    with st.form("custom_test"):
        c1, c2 = st.columns(2)
        with c1:
            custom_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, key="ct")
            custom_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0, key="cm")
        with c2:
            custom_humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, key="ch")
            custom_light = st.slider("Light Intensity", 0, 1000, 500, key="cl")
        
        if st.form_submit_button("Analyze Custom Conditions"):
            custom_prediction = predict_irrigation(custom_temp, custom_moisture, custom_humidity, custom_light)
            
            # Display detailed analysis
            st.write("### 🔍 Detailed Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Input Values:**")
                st.write(f"🌡️ Temperature: {custom_temp}°C")
                st.write(f"💧 Soil Moisture: {custom_moisture}%")
                st.write(f"💨 Humidity: {custom_humidity}%")
                st.write(f"💡 Light: {custom_light}")
            
            with col2:
                st.write("**AI Factors:**")
                for factor, value in custom_prediction['factors'].items():
                    st.write(f"• {factor}: {value:.3f}")
                st.write(f"**Final Score:** {custom_prediction['score']:.3f}")
            
            if custom_prediction['irrigation_decision'] == 'yes':
                st.error(f"🚨 **IRRIGATION DECISION: YES** (Confidence: {custom_prediction['confidence_level']:.1%})")
            else:
                st.success(f"✅ **IRRIGATION DECISION: NO** (Confidence: {custom_prediction['confidence_level']:.1%})")

# System Status
st.markdown("---")
st.header("🔧 System Status & Configuration")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.subheader("🌐 Connectivity")
    if supabase_client:
        st.success("✅ Supabase Connected")
    else:
        st.error("❌ Supabase Offline")
    
    if latest_data:
        st.success("✅ ESP32 Online")
    else:
        st.warning("⚠️ ESP32 Offline")

with status_col2:
    st.subheader("📊 Data Flow")
    if historical_data:
        data_count = len(historical_data)
        st.success(f"✅ {data_count} Records")
    else:
        st.warning("⚠️ No Data")

with status_col3:
    st.subheader("🤖 AI System")
    st.success("✅ Model Active")
    st.caption("Advanced Rule-based AI")

with status_col4:
    st.subheader("⚙️ Settings")
    if st.button("🔄 Force Refresh"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("### 💧 Smart Irrigation System | 🤖 AI-Powered Decisions")
st.markdown("*Real-time monitoring and intelligent irrigation predictions*")
