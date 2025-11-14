# ---------------------- SIMULATION SECTION ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔬 Simulation Lab</div>', unsafe_allow_html=True)
st.markdown('<p class="status-text">Test how different environmental conditions affect irrigation needs</p>', unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([1.2, 1.2])

# Initialize simulation result
sim_result = None

with sim_col1:
    st.markdown('<div class="simulation-controls">', unsafe_allow_html=True)
    st.write("**Adjust environmental parameters:**")
    
    # Use Streamlit's native slider styling
    sim_temp = st.slider(
        "🌡️ Temperature (°C)", 
        min_value=0.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.5,
        key="sim_temp"
    )
    sim_soil = st.slider(
        "💧 Soil Moisture (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=1.0,
        key="sim_soil"
    )
    sim_hum = st.slider(
        "🌫️ Air Humidity (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=60.0, 
        step=1.0,
        key="sim_hum"
    )
    sim_light = st.slider(
        "☀️ Light Intensity", 
        min_value=0, 
        max_value=1500, 
        value=500, 
        step=10,
        key="sim_light"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Show current simulation values in a nice box
    st.markdown("""
    <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; border: 1px solid #bbf7d0; margin-top: 1rem;">
        <div style="text-align: center; margin-bottom: 0.5rem; font-weight: 600; color: #166534;">Current Simulation Values</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #4d7c0f;">Temperature</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{}°C</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #4d7c0f;">Soil Moisture</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #4d7c0f;">Humidity</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #4d7c0f;">Light</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #166534;">{}</div>
            </div>
        </div>
    </div>
    """.format(sim_temp, sim_soil, sim_hum, sim_light), unsafe_allow_html=True)

    if artifacts is None:
        st.warning("🤖 The AI model is not loaded. Simulation features are currently unavailable.")
    else:
        sim_result = model_predict(sim_temp, sim_soil, sim_hum, sim_light, crop_type="tomato")

        if sim_result is None:
            st.error("❌ Could not compute simulation with these values.")
        else:
            sim_decision = sim_result["irrigation_prediction"]
            sim_conf = sim_result["confidence_level"]

            if sim_decision == "yes":
                st.success(f"💦 **Simulated Advice: Water Recommended**")
                st.write(f"With these conditions, the model suggests watering with **{sim_conf:.0%} confidence**")
            else:
                st.info(f"✅ **Simulated Advice: No Water Needed**")
                st.write(f"Current simulated conditions don't require watering (**{sim_conf:.0%} confidence**)")

with sim_col2:
    st.markdown('<div class="simulation-controls">', unsafe_allow_html=True)
    st.write("**🌿 Simulated Plant Response**")

    if artifacts is not None and sim_result is not None:
        sim_decision = sim_result["irrigation_prediction"]
        
        if sim_soil > 70 and sim_decision == "no":
            sim_state_label = "Thriving"
            sim_emoji = "🌿"
            sim_note = "Perfect conditions! The plant would be lush and vibrant with optimal soil moisture."
            sim_status_class = "plant-status-healthy"
        elif sim_soil < 40 or sim_decision == "yes":
            sim_state_label = "Stressed"
            sim_emoji = "🥀"
            sim_note = "The plant would show signs of dehydration. Leaves might droop and soil feels dry."
            sim_status_class = "plant-status-attention"
        else:
            sim_state_label = "Stable"
            sim_emoji = "🌱"
            sim_note = "The plant would be growing steadily but could benefit from improved conditions."
            sim_status_class = "plant-status-stable"

        st.markdown(f"""
        <div class="metric-card {sim_status_class}" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{sim_emoji}</div>
            <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; color: inherit;">{sim_state_label}</div>
            <div style="font-size: 0.9rem; color: inherit;">{sim_note}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("**Simulated Environment:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Soil", f"{sim_soil}%")
            st.metric("Light", f"{sim_light}")
        with col2:
            st.metric("Temp", f"{sim_temp}°C")
            st.metric("Humidity", f"{sim_hum}%")
    else:
        st.info("🎛️ Adjust the sliders on the left to see how different conditions affect plant health and irrigation needs.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
