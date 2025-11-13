import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from supabase import create_client, Client
import plotly.express as px

# ----------------------------- #
# 🌍 CONFIGURATION
# ----------------------------- #
THINGSPEAK_CHANNEL_ID = "3125494"  # Replace with your own channel ID
THINGSPEAK_API_KEY = "YOUR_THINGSPEAK_API_KEY"  # Optional (read key)
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Streamlit page config
st.set_page_config(page_title="🌱 TomoGrow Dashboard", layout="wide")

st.title("🌿 TomoGrow Smart Garden Dashboard")
st.caption("Real-time ESP32 sensor data visualization from ThingSpeak + Supabase")

# ----------------------------- #
# ⚙️ DATA FETCHING
# ----------------------------- #
def get_thingspeak_data():
    """Fetch the last 50 entries from ThingSpeak."""
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?results=50"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()["feeds"]

        records = []
        for d in data:
            records.append({
                "timestamp": datetime.strptime(d["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                "temperature": float(d.get("field1") or 0),
                "humidity": float(d.get("field2") or 0),
                "soil_moisture": float(d.get("field3") or 0),
                "light_intensity": float(d.get("field4") or 0),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp")
        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()


def save_to_supabase(df):
    """Insert new records to Supabase."""
    try:
        data_to_insert = df.to_dict(orient="records")
        if data_to_insert:
            supabase.table("sensor_data").insert(data_to_insert).execute()
    except Exception as e:
        st.warning(f"⚠️ Could not save to Supabase: {e}")


# ----------------------------- #
# 🧠 DASHBOARD UI
# ----------------------------- #
placeholder = st.empty()

# Sidebar refresh controller
st.sidebar.header("🔁 Auto Refresh Settings")
interval = st.sidebar.slider("Refresh interval (seconds)", 10, 60, 15)

while True:
    df = get_thingspeak_data()
    if df.empty:
        st.warning("No data found from ThingSpeak yet.")
        time.sleep(interval)
        st.rerun()

    latest = df.iloc[-1]

    with placeholder.container():
        st.subheader("🌡️ Latest Sensor Readings")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Temperature 🌞", f"{latest.temperature:.1f} °C")

        with col2:
            st.metric("Humidity 💧", f"{latest.humidity:.1f} %")

        with col3:
            st.metric("Soil Moisture 🌱", f"{latest.soil_moisture:.1f} %")

        with col4:
            st.metric("Light Intensity 💡", f"{latest.light_intensity:.1f} lux")

        # Historical charts
        st.divider()
        st.subheader("📊 Historical Trends")

        col1, col2 = st.columns(2)

        with col1:
            fig_temp = px.line(df, x="timestamp", y="temperature", title="🌡️ Temperature Over Time")
            st.plotly_chart(fig_temp, use_container_width=True)

            fig_soil = px.line(df, x="timestamp", y="soil_moisture", title="🌱 Soil Moisture Over Time")
            st.plotly_chart(fig_soil, use_container_width=True)

        with col2:
            fig_hum = px.line(df, x="timestamp", y="humidity", title="💧 Humidity Over Time")
            st.plotly_chart(fig_hum, use_container_width=True)

            fig_light = px.line(df, x="timestamp", y="light_intensity", title="💡 Light Intensity Over Time")
            st.plotly_chart(fig_light, use_container_width=True)

        # Save to Supabase (optional)
        save_to_supabase(df)

        # Auto-refresh info
        st.info(f"⏳ Auto-refreshing every {interval} seconds...")
        time.sleep(interval)
        st.rerun()
