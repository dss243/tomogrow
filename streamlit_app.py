import requests
from supabase import create_client
import os

# ---------- ThingSpeak config ----------
THINGSPEAK_CHANNEL_ID = "3125494"
THINGSPEAK_READ_API_KEY = "QAQOE30K5W5UTZTU"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"

# ---------- Supabase config ----------
SUPABASE_URL = "https://rcptkfgiiwgskbegdcih.supabase.co"
# IMPORTANT: use a service role key or a key with insert permission on sensor_data
SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_latest_from_thingspeak():
  params = {
      "api_key": THINGSPEAK_READ_API_KEY,
      "results": 1
  }
  r = requests.get(THINGSPEAK_URL, params=params, timeout=10)
  r.raise_for_status()
  feeds = r.json()["feeds"]
  if not feeds:
    return None
  return feeds[0]

def sync_latest():
  latest = fetch_latest_from_thingspeak()
  if latest is None:
    print("No ThingSpeak data yet")
    return

  # Map ThingSpeak fields to our Supabase columns
  temperature     = float(latest["field1"]) if latest["field1"] is not None else None
  humidity        = float(latest["field2"]) if latest["field2"] is not None else None
  soil_moisture   = float(latest["field3"]) if latest["field3"] is not None else None
  light_intensity = float(latest["field4"]) if latest["field4"] is not None else None

  row = {
      "device_id": "ESP32_TOMOGROW_001",
      "temperature":   temperature,
      "humidity":      humidity,
      "soil_moisture": soil_moisture,
      "light_intensity": light_intensity
      # Optionally store created_at using ThingSpeak 'created_at'
      # "created_at": latest["created_at"]
  }

  print("Inserting into Supabase:", row)
  supabase.table("sensor_data").insert(row).execute()

if __name__ == "__main__":
  sync_latest()
