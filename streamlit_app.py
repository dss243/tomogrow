import streamlit as st
from supabase import create_client
import pickle
import numpy as np
import os
import pandas as pd

# ------------------ Supabase config ------------------
SUPABASE_URL = "https://ragapkdlgtpmumwlzphs.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2Fwa2RsZ3RwbXVtd2x6cGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2MTYwMDMsImV4cCI6MjA3ODE5MjAwM30.OQj-NFgd6KaDKL1BobPgLOKTCYDFmqw8KnqQFzkFWKo"
DEVICE_ID = "ESP32_TOMOGROW_001"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

supabase_client = init_supabase()

def get_latest_data():
    if not supabase_client:
        return None
    response = (
        supabase_client
        .table("sensor_data")
        .select("*")
        .eq("device_id", DEVICE_ID)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    data = response.data or []
    return data[0] if data else None

def get_history(limit: int = 100):
    if not supabase_client:
        return None
    response = (
        supabase_client
        .table("sensor_data")
        .select("*")
        .eq("device_id", DEVICE_ID)
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    data = response.data or []
    if not data:
        return None
    df = pd.DataFrame(data)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
    return df

# ------------------ UI setup (short) ------------------
st.set_page_config(page_title="TomoGrow – Smart Irrigation", layout="wide")

latest_data = get_latest_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Live Field Snapshot")
    if latest_data:
        temperature = float(latest_data.get("temperature", 0))
        humidity = float(latest_data.get("humidity", 0))
        soil_moisture = float(latest_data.get("soil_moisture", 0))
        light_intensity = float(latest_data.get("light_intensity", 0))
        timestamp = latest_data.get("created_at", "")

        st.metric("Temperature (°C)", f"{temperature:.1f}")
        st.metric("Humidity (%)", f"{humidity:.1f}")
        st.metric("Soil Moisture (%)", f"{soil_moisture:.1f}")
        st.metric("Light", int(light_intensity))
        st.caption(f"Last update: {timestamp}")
    else:
        st.info("No sensor data yet.")

with col2:
    st.subheader("📈 Sensor History")
    df_hist = get_history(limit=80)
    if df_hist is not None:
        metric_choice = st.selectbox(
            "Metric",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=2,
        )
        st.line_chart(df_hist.set_index("created_at")[metric_choice])
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No historical data yet.")
