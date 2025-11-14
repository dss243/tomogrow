from supabase import create_client

SUPABASE_URL = "https://ragapkdlgtpmumwlzphs.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2Fwa2RsZ3RwbXVtd2x6cGhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2MTYwMDMsImV4cCI6MjA3ODE5MjAwM30.OQj-NFgd6KaDKL1BobPgLOKTCYDFmqw8KnqQFzkFWKo"

@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return client
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        return None

supabase_client = init_supabase()
