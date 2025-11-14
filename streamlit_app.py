import streamlit as st

st.set_page_config(page_title="Test Supabase Secrets", page_icon="🔑")

st.title("🔑 Supabase Secrets Test")

st.write("Trying to read SUPABASE_URL and SUPABASE_ANON_KEY from st.secrets...")

# Try to read secrets and show them (partially) for debugging
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    st.success("✅ Found SUPABASE_URL and SUPABASE_ANON_KEY in st.secrets.")
    st.write(f"SUPABASE_URL = {url}")
    st.write(f"SUPABASE_ANON_KEY (first 10 chars) = {key[:10]}...")
except Exception as e:
    st.error(f"Supabase init failed: {e}")
