import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import supabase
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
import os

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

# Load your trained ML model with better error handling
@st.cache_resource
def load_ml_model():
    model_paths = [
        'fast_tomato_irrigation_model.pkl',
        './fast_tomato_irrigation_model.pkl',
        'model/fast_tomato_irrigation_model.pkl'
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    model_data = pickle.load(file)
                
                st.success(f"✅ ML Model Loaded from: {model_path}")
                return model_data
        except Exception as e:
            st.error(f"❌ Failed to load model from {model_path}: {e}")
            continue
    
    st.warning("ML Model File Not Found - Using rule-based system")
    return None

supabase_client = init_supabase()
ml_model_data = load_ml_model()

def predict_irrigation_ml(temperature, soil_moisture, humidity, light_intensity, crop_type="tomato"):
    """Use your trained Random Forest model for prediction"""
    if ml_model_data is None:
        return predict_irrigation_rules(temperature, soil_moisture, humidity, light_intensity)
    
    try:
        # Get model components
        model = ml_model_data['model']
        scaler = ml_model_data['scaler']
        crop_encoder = ml_model_data.get('crop_encoder')
        feature_names = ml_model_data.get('feature_names', [])
        
        # Encode crop type using the same encoder from training
        if crop_encoder is not None:
            try:
                # Transform crop type to encoded value
                crop_encoded = crop_encoder.transform([crop_type])[0]
            except:
                # If crop type not in encoder, use default (tomato)
                st.sidebar.warning(f"Crop type '{crop_type}' not in encoder, using default")
                crop_encoded = 0  # Default to tomato
        else:
            # If no encoder, use simple mapping
            crop_mapping = {'tomato': 0, 'cucumber': 1, 'pepper': 2, 'lettuce': 3}
            crop_encoded = crop_mapping.get(crop_type.lower(), 0)
        
        # Prepare features in EXACT same order as training
        features = np.array([[
            temperature,      # Temperature
            soil_moisture,    # Soil_Moisture  
            humidity,         # Humidity
            light_intensity,  # Light_Intensity
            crop_encoded      # Crop_Type_encoded
        ]])
        
        # Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get confidence
        confidence = np.max(probabilities)
        
        # Map prediction to decision (using pump_encoder if available)
        pump_encoder = ml_model_data.get('pump_encoder')
        if pump_encoder is not None:
            decision = pump_encoder.inverse_transform([prediction])[0]
        else:
            decision = "yes" if prediction == 1 else "no"
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_') and feature_names:
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            'irrigation_prediction': decision,
            'irrigation_decision': decision,
            'confidence_level': round(confidence, 4),
            'soil_moisture_level': soil_moisture,
            'model_used': 'RandomForest',
            'probabilities': {
                'no': probabilities[0],
                'yes': probabilities[1]
            },
            'feature_importance': feature_importance,
            'crop_type': crop_type,
            'crop_encoded': crop_encoded
        }
        
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        # Fallback to rule-based system
        return predict_irrigation_rules(temperature, soil_moisture, humidity, light_intensity)

def predict_irrigation_rules(temperature, soil_moisture, humidity, light_intensity):
    """Fallback rule-based system"""
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
        'soil_moisture_level': soil_moisture,
        'model_used': 'RuleBased'
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
    return None

def create_timestamp_column(df):
    """Create proper timestamp for charts"""
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = df.sort_values('id').reset_index(drop=True)
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=len(df)),
            periods=len(df),
            freq='30S'
        )
    return df

# Dashboard UI
st.title("🌱 Smart Irrigation AI Dashboard")
st.markdown("---")

# Model Information Sidebar
st.sidebar.header("🤖 ML Model Information")
if ml_model_data:
    st.sidebar.success("**Random Forest Classifier**")
    st.sidebar.write(f"**Accuracy:** {ml_model_data.get('training_accuracy', '100.0%')}")
    st.sidebar.write(f"**Model Type:** {ml_model_data.get('model_type', 'RandomForest')}")
    
    if 'feature_names' in ml_model_data:
        st.sidebar.subheader("📋 Training Features")
        for i, feature in enumerate(ml_model_data['feature_names']):
            st.sidebar.write(f"{i+1}. {feature}")
    
    # Crop type selection
    st.sidebar.subheader("🌱 Crop Selection")
    crop_type = st.sidebar.selectbox(
        "Select crop type:",
        ["tomato", "cucumber", "pepper", "lettuce"],
        index=0
    )
    st.sidebar.info(f"Current crop: **{crop_type}**")
else:
    st.sidebar.warning("Using Rule-Based System")
    crop_type = "tomato"

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
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        temperature = latest_data.get('temperature', 0)
        if temperature is not None:
            st.metric("🌡️ Temperature", f"{float(temperature):.1f}°C")
        else:
            st.metric("🌡️ Temperature", "N/A")
    
    with col2:
        humidity = latest_data.get('humidity', 0)
        if humidity is not None:
            st.metric("💧 Humidity", f"{float(humidity):.1f}%")
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
    
    with col5:
        st.metric("🌿 Crop Type", crop_type.title())

    # AI Prediction with ML Model
    if all(key in latest_data and latest_data[key] is not None for key in ['temperature', 'soil_moisture', 'humidity', 'light_intensity']):
        prediction = predict_irrigation_ml(
            float(latest_data['temperature']),
            float(latest_data['soil_moisture']),
            float(latest_data['humidity']),
            int(latest_data['light_intensity']),
            crop_type
        )
        
        # Enhanced prediction display
        st.subheader("🎯 AI Irrigation Decision")
        
        pred_col1, pred_col2 = st.columns([2, 1])
        
        with pred_col1:
            if prediction['irrigation_decision'] == 'yes':
                st.error(f"🚨 **IRRIGATION NEEDED**")
            else:
                st.success(f"✅ **NO IRRIGATION NEEDED**")
            
            # Confidence meter
            confidence = prediction['confidence_level']
            st.write(f"**Confidence:** {confidence:.1%}")
            st.progress(float(confidence))
            
            # Model used
            st.write(f"**Model:** {prediction['model_used']}")
            st.write(f"**Crop:** {prediction['crop_type']}")
        
        with pred_col2:
            with st.expander("📊 Prediction Details"):
                if 'probabilities' in prediction:
                    st.write("**Class Probabilities:**")
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("No Irrigation", f"{prediction['probabilities']['no']:.1%}")
                    with col_prob2:
                        st.metric("Yes Irrigation", f"{prediction['probabilities']['yes']:.1%}")
                
                if prediction.get('feature_importance'):
                    st.write("**Feature Importance:**")
                    for feature, importance in prediction['feature_importance'].items():
                        st.write(f"• {feature}: {importance:.3f}")
                
                st.write(f"**Crop Encoded:** {prediction.get('crop_encoded', 'N/A')}")

    else:
        st.warning("⚠️ Incomplete data for AI prediction")

else:
    st.warning("📡 Waiting for ESP32 data...")

# Model Testing Section
st.markdown("---")
st.header("🧪 Test ML Model with Different Crops")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Quick Scenarios")
    
    scenario = st.selectbox("Choose scenario:", 
                           ["Normal Day", "Hot & Dry", "Cool & Wet", "Extreme Dry", "Optimal Conditions"])
    
    test_crop = st.selectbox("Crop for test:", ["tomato", "cucumber", "pepper", "lettuce"], key="test_crop")
    
    if st.button("Run ML Prediction"):
        scenarios = {
            "Normal Day": (25, 60, 65, 500),
            "Hot & Dry": (35, 30, 40, 800),
            "Cool & Wet": (18, 85, 75, 300),
            "Extreme Dry": (30, 15, 35, 900),
            "Optimal Conditions": (22, 65, 60, 600)
        }
        
        temp, moisture, hum, light = scenarios[scenario]
        prediction = predict_irrigation_ml(temp, moisture, hum, light, test_crop)
        
        st.write(f"**Scenario:** {scenario}")
        st.write(f"**Crop:** {test_crop.title()}")
        st.write(f"**Conditions:** {temp}°C, {moisture}% soil, {hum}% humidity, {light} light")
        
        if prediction['irrigation_decision'] == 'yes':
            st.error(f"🚨 ML DECISION: IRRIGATION NEEDED")
        else:
            st.success(f"✅ ML DECISION: NO IRRIGATION NEEDED")
        
        st.write(f"**Confidence:** {prediction['confidence_level']:.1%}")
        st.write(f"**Model:** {prediction['model_used']}")

with col2:
    st.subheader("Custom ML Test")
    with st.form("custom_ml_test"):
        c1, c2 = st.columns(2)
        with c1:
            custom_temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, key="ml_temp")
            custom_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0, key="ml_moisture")
            custom_crop = st.selectbox("Crop Type", ["tomato", "cucumber", "pepper", "lettuce"], key="form_crop")
        with c2:
            custom_humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, key="ml_humidity")
            custom_light = st.slider("Light Intensity", 0, 1000, 500, key="ml_light")
        
        if st.form_submit_button("Run ML Analysis"):
            prediction = predict_irrigation_ml(custom_temp, custom_moisture, custom_humidity, custom_light, custom_crop)
            
            st.write("### 🔬 ML Analysis Results")
            
            if prediction['irrigation_decision'] == 'yes':
                st.error(f"🚨 **ML DECISION: IRRIGATION NEEDED**")
            else:
                st.success(f"✅ **ML DECISION: NO IRRIGATION NEEDED**")
            
            st.write(f"**Confidence Level:** {prediction['confidence_level']:.1%}")
            st.write(f"**Model Used:** {prediction['model_used']}")
            st.write(f"**Crop Type:** {prediction['crop_type'].title()}")
            
            if 'probabilities' in prediction:
                st.write("**Class Probabilities:**")
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("No Irrigation", f"{prediction['probabilities']['no']:.1%}")
                with col_prob2:
                    st.metric("Yes Irrigation", f"{prediction['probabilities']['yes']:.1%}")

# System Status
st.markdown("---")
st.header("🔧 System Status")

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
    st.subheader("🤖 AI System")
    if ml_model_data:
        st.success("✅ ML Model Active")
        st.caption(f"Random Forest (5 features)")
    else:
        st.warning("⚠️ Rule-Based System")

with status_col3:
    st.subheader("📊 Data Flow")
    if latest_data:
        st.success("✅ Live Data")
    else:
        st.warning("⚠️ No Data")

with status_col4:
    st.subheader("🌱 Current Crop")
    st.success(f"✅ {crop_type.title()}")

# Footer
st.markdown("---")
st.markdown("### 🌱 Smart Irrigation AI System | 🧠 ML-Powered Decisions")
st.markdown("*Real-time monitoring with trained Random Forest model*")
