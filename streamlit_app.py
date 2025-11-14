# ---------------------- RIGHT COLUMN ----------------------
with col2:
    # Sensor History & Trends
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Sensor History & Trends</div>', unsafe_allow_html=True)
    
    # Organize controls in columns
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        points = st.slider(
            "Data points",
            min_value=20,
            max_value=200,
            value=80,
            step=20,
            help="Number of historical points to display"
        )
    
    with control_col2:
        metric_choice = st.selectbox(
            "Metric",
            ["temperature", "humidity", "soil_moisture", "light_intensity"],
            index=2,
            format_func=lambda x: {
                "temperature": "🌡️ Temperature",
                "humidity": "💧 Humidity", 
                "soil_moisture": "🌱 Soil Moisture",
                "light_intensity": "☀️ Light Intensity"
            }[x]
        )
    
    df_hist = get_history(limit=points)
    
    if df_hist is not None:
        # Display chart
        st.markdown("**Live Trend**")
        st.line_chart(
            df_hist.set_index("created_at")[metric_choice],
            height=320
        )
        
        # Recent data table
        st.markdown("**Recent Measurements**")
        st.dataframe(
            df_hist[["created_at", "temperature", "humidity", "soil_moisture", "light_intensity"]].tail(6),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("📊 No historical data available yet. Data will accumulate over time.")
    
    st.markdown("</div>", unsafe_allow_html=True)
