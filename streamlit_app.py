# streamlit_mqtt_to_supabase.py
import streamlit as st
import threading
import json
import time
import requests
import pandas as pd
from paho.mqtt import client as mqtt_client

# ---------------- CONFIG ----------------
MQTT_BROKER = "mqtt.wokwi.cloud"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/tomogrow/ESP32_TOMOGROW_001"  # must match ESP32 topic

# Supabase REST settings (use your project's URL & ANON key or service key for dev)
SUPABASE_URL = "https://ragapkdlgtpmumwlzphs.supabase.co/rest/v1/sensor_data"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2Fwa2RsZ3RwbXVtd2x6cGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2MTYwMDMsImV4cCI6MjA3ODE5MjAwM30.OQj-NFgd6KaDKL1BobPgLOKTCYDFmqw8KnqQFzkFWKo"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# ---------------- MQTT -> Supabase bridge ----------------
st.set_page_config(page_title="MQTT→Supabase Bridge", layout="wide")

st.title("ESP32 → MQTT → Streamlit → Supabase bridge")

# Status area
status_placeholder = st.empty()
log_box = st.empty()

# Use a simple thread-safe list for logs
log_lines = []
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)
    if len(log_lines) > 200:
        log_lines.pop(0)
    # update UI
    log_box.text("\n".join(log_lines[-20:]))

# Supabase insert function
def insert_into_supabase(row: dict):
    try:
        r = requests.post(SUPABASE_URL, headers=HEADERS, json=row, timeout=10)
        if not r.ok:
            log(f"Supabase insert failed: {r.status_code} {r.text}")
            return False
        # r.json() contains the inserted row if 'Prefer: return=representation' is used
        log(f"Inserted into Supabase (id maybe): {r.status_code}")
        return True
    except Exception as e:
        log(f"Supabase request exception: {e}")
        return False

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        log(f"Subscribed to: {MQTT_TOPIC}")
    else:
        log(f"Failed to connect MQTT, rc={rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        log(f"MQTT message on {msg.topic}: {payload}")
        data = json.loads(payload)
        # Ensure fields exist and build row
        row = {
            "device_id": data.get("device_id", "unknown"),
            "temperature": float(data.get("temperature")) if data.get("temperature") is not None else None,
            "humidity": float(data.get("humidity")) if data.get("humidity") is not None else None,
            "soil_moisture": float(data.get("soil_moisture")) if data.get("soil_moisture") is not None else None,
            "light_intensity": int(data.get("light_intensity")) if data.get("light_intensity") is not None else None
        }
        # Insert into Supabase
        ok = insert_into_supabase(row)
        if ok:
            log("Row saved to Supabase")
        else:
            log("Failed to save row")
    except Exception as e:
        log(f"Error processing MQTT message: {e}")

# Start MQTT client in background thread
def start_mqtt_client():
    client_id = f"streamlit-bridge-{int(time.time())}"
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        log(f"MQTT connect error: {e}")
        return

    # loop_forever blocks; run it in this thread
    try:
        client.loop_forever()
    except Exception as e:
        log(f"MQTT loop stopped: {e}")

# Run MQTT thread once
if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = True
    t = threading.Thread(target=start_mqtt_client, daemon=True)
    t.start()
    log("Starting MQTT background thread...")

# ---------------- UI: show latest rows from Supabase ----------------
st.markdown("### Latest sensor rows in Supabase")
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Refresh from Supabase"):
        pass  # refresh handled below

    st.write("Controls")
    st.write("MQTT Broker: " + MQTT_BROKER)
    st.write("MQTT Topic: " + MQTT_TOPIC)

with col2:
    # Fetch recent rows
    def fetch_recent(limit=50):
        try:
            params = {"select": "*", "limit": limit, "order": "id.desc"}
            # build query string manually
            q = SUPABASE_URL + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
            r = requests.get(q, headers=HEADERS, timeout=10)
            if not r.ok:
                st.error(f"Error fetching from Supabase: {r.status_code} {r.text}")
                return None
            data = r.json()
            if not data:
                return None
            df = pd.DataFrame(data)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            return df
        except Exception as e:
            st.error(f"Exception fetching Supabase: {e}")
            return None

    df = fetch_recent(limit=50)
    if df is not None:
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("No rows found in Supabase or fetch error (check logs above).")

# Show logs at bottom
st.markdown("---")
st.markdown("#### Bridge log (latest)")
log_box.text("\n".join(log_lines[-20:]))

# End of app
