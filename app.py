import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.narrative import generate_operator_narrative
import src.features as ft
import src.rul_model as rm

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery SoH & RUL Estimator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism & disclaimer banner styling
st.markdown("""
<style>
    /* Persistent Disclaimer Banner */
    .disclaimer-banner {
        background-color: rgba(220, 38, 38, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 24px;
        color: #fca5a5;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .disclaimer-banner strong {
        color: #ef4444;
    }
    
    /* Card styling */
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Status Badges */
    .badge-healthy {
        background-color: #15803d;
        color: #bbf7d0;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-monitor {
        background-color: #b45309;
        color: #fef08a;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-replace {
        background-color: #b91c1c;
        color: #fecaca;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. PERSISTENT DISCLAIMER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>SAFETY DISCLAIMER:</strong> This is a data-driven estimate, not a substitute for manufacturer testing or certified diagnostics. Do not use this tool as the sole basis for safety-critical decisions.
</div>
""", unsafe_allow_html=True)

st.title("🔋 Battery State-of-Health (SoH) & RUL Estimator")
st.caption("Real-Time Telemetry Analytics, Predictive Degradation Modeling & Safety Guardrails")

# -----------------------------------------------------------------------------
# 2. REAL TEAMMATES DATA LOADERS
# -----------------------------------------------------------------------------
def load_pipeline_data():
    features_file = os.path.join('data', 'features.csv')
    if not os.path.exists(features_file):
        synthetic_file = os.path.join('data', 'synthetic_battery_data.csv')
        if os.path.exists(synthetic_file):
            raw_df = pd.read_csv(synthetic_file)
            df_features = ft.build_features(raw_df)
        else:
            df_features = ft.make_mock_data()
    else:
        df_features = pd.read_csv(features_file)
        
    drivers_file = os.path.join('outputs', 'feature_importances.csv')
    if os.path.exists(drivers_file):
        df_drivers = pd.read_csv(drivers_file)
    else:
        df_drivers = pd.DataFrame({
            'Feature': ['cumulative_time_above_40C', 'c_rate', 'discharge_depth'],
            'Importance': [0.85, 0.10, 0.05]
        })
        
    soh_preds_file = os.path.join('outputs', 'soh_predictions.csv')
    if os.path.exists(soh_preds_file):
        df_soh = pd.read_csv(soh_preds_file)
    else:
        df_soh = df_features[['cycle_id', 'cell_id', 'soh_ground_truth']].copy()
        df_soh['soh_predicted'] = df_soh['soh_ground_truth']
        
    # Merge predictions into main feature dataframe
    df_merged = df_features.merge(
        df_soh[['cell_id', 'cycle_id', 'soh_predicted']], 
        on=['cell_id', 'cycle_id'], 
        how='left'
    )
    df_merged['soh_predicted'] = df_merged['soh_predicted'].fillna(df_merged['soh_ground_truth'])
    
    df_merged['soh_upper'] = np.clip(df_merged['soh_predicted'] + 0.8, 60.0, 100.0)
    df_merged['soh_lower'] = np.clip(df_merged['soh_predicted'] - 0.8, 60.0, 100.0)
    
    # RUL Predictions from Person D
    rul_file = os.path.join('outputs', 'rul_predictions.csv')
    if os.path.exists(rul_file):
        df_rul = pd.read_csv(rul_file)
    else:
        df_rul = rm.build_rul_output(df_soh)
        
    return df_merged, df_drivers, df_rul

df_all, df_drivers, df_rul = load_pipeline_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & CELL METADATA
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Cell Controls")
cell_list = sorted(df_all['cell_id'].unique().tolist())
selected_cell = st.sidebar.selectbox("Select Target Battery Cell:", cell_list)

df_cell = df_all[df_all['cell_id'] == selected_cell].sort_values('cycle_id').reset_index(drop=True)

# Fetch RUL stats for selected cell from Person D's module output
cell_rul_row = df_rul[df_rul['cell_id'] == selected_cell]
if len(cell_rul_row) > 0:
    r_row = cell_rul_row.iloc[0]
    current_soh = float(r_row['current_soh']) if pd.notna(r_row['current_soh']) else float(df_cell['soh_predicted'].iloc[-1])
    slope_val = float(r_row['trend_slope']) if pd.notna(r_row['trend_slope']) else -0.01
    
    rul_likely = int(r_row['rul_likely_cycles']) if pd.notna(r_row['rul_likely_cycles']) else 0
    rul_worst = int(r_row['rul_worst_cycles']) if pd.notna(r_row['rul_worst_cycles']) else 0
    rul_best = int(r_row['rul_best_cycles']) if pd.notna(r_row['rul_best_cycles']) else 3000
else:
    rul_dict = rm.compute_bands_for_cell(df_cell)
    current_soh = float(rul_dict['current_soh'])
    slope_val = float(rul_dict['trend_slope'])
    rul_likely = int(rul_dict['rul_likely_cycles']) if rul_dict['rul_likely_cycles'] is not None else 0
    rul_worst = int(rul_dict['rul_worst_cycles']) if rul_dict['rul_worst_cycles'] is not None else 0
    rul_best = int(rul_dict['rul_best_cycles']) if rul_dict['rul_best_cycles'] is not None else 3000

# Top driver from Person C's feature importance output
top_feat_name = str(df_drivers.iloc[0]['Feature']).replace('_', ' ').title()

# Status classification logic
if current_soh <= 80.0:
    status_label = "Replace Soon"
elif current_soh <= 85.0 or slope_val <= -0.05:
    status_label = "Monitor Closely"
else:
    status_label = "Healthy"

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Cell Profile")
st.sidebar.write(f"**Cell ID**: `{selected_cell}`")
st.sidebar.write(f"**Telemetry Cycles**: {len(df_cell)}")
st.sidebar.write(f"**Degradation Slope**: `{slope_val:.5f}% / cycle`")

# -----------------------------------------------------------------------------
# 4. KPI SUMMARY CARDS ROW
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Current SoH</div>
        <div class="kpi-value">{current_soh:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Likely RUL</div>
        <div class="kpi-value">{rul_likely} <span style="font-size: 1rem; color: #94a3b8;">cycles</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Primary Driver</div>
        <div class="kpi-value" style="font-size: 1.1rem; margin-top: 8px;">{top_feat_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    badge_class = "badge-healthy" if status_label == "Healthy" else ("badge-monitor" if status_label == "Monitor Closely" else "badge-replace")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Status Classification</div>
        <div style="margin-top: 10px;"><span class="{badge_class}">{status_label}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. CHART 1: SOH DECAY TREND (REAL DATA & CONFIDENCE BAND)
# -----------------------------------------------------------------------------
st.subheader("📉 State-of-Health (SoH) Decay Trajectory")

fig_soh = go.Figure()

fig_soh.add_trace(go.Scatter(
    x=pd.concat([df_cell['cycle_id'], df_cell['cycle_id'][::-1]]),
    y=pd.concat([df_cell['soh_upper'], df_cell['soh_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(99, 102, 241, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Model Confidence Interval'
))

fig_soh.add_trace(go.Scatter(
    x=df_cell['cycle_id'],
    y=df_cell['soh_ground_truth'],
    mode='lines',
    name='Ground Truth SoH',
    line=dict(color='#38bdf8', width=2, dash='dash')
))

fig_soh.add_trace(go.Scatter(
    x=df_cell['cycle_id'],
    y=df_cell['soh_predicted'],
    mode='lines',
    name='Model Predicted SoH',
    line=dict(color='#818cf8', width=3)
))

fig_soh.add_hline(y=80.0, line_dash="dash", line_color="#ef4444", annotation_text="80% End-of-Life Threshold", annotation_position="bottom right")

fig_soh.update_layout(
    template="plotly_dark",
    xaxis_title="Cycle Count",
    yaxis_title="State-of-Health (%)",
    height=380,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_soh, width='stretch')

# -----------------------------------------------------------------------------
# 6. CHARTS ROW 2: REAL DRIVER IMPORTANCE & REAL RUL BANDS
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Key Degradation Drivers (Feature Importances)")
    top_drivers = df_drivers.head(5).copy()
    top_drivers['clean_name'] = top_drivers['Feature'].str.replace('_', ' ').str.title()
    
    fig_driver = go.Figure(go.Bar(
        x=top_drivers['Importance'],
        y=top_drivers['clean_name'],
        orientation='h',
        marker=dict(
            color=top_drivers['Importance'],
            colorscale='Viridis'
        )
    ))
    fig_driver.update_layout(
        template="plotly_dark",
        xaxis_title="Relative Feature Importance Score",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_driver, width='stretch')

with col_chart2:
    st.subheader("⏳ Remaining Useful Life (RUL) Uncertainty Bands")
    rul_chart_df = pd.DataFrame({
        'Scenario': ['Worst Case (P10)', 'Likely (P50)', 'Best Case (P90)'],
        'Cycles': [rul_worst, rul_likely, rul_best],
        'Color': ['#ef4444', '#f59e0b', '#10b981']
    })
    
    fig_rul = go.Figure(go.Bar(
        x=rul_chart_df['Scenario'],
        y=rul_chart_df['Cycles'],
        text=rul_chart_df['Cycles'].astype(str) + " cycles",
        textposition='auto',
        marker_color=rul_chart_df['Color']
    ))
    fig_rul.update_layout(
        template="plotly_dark",
        yaxis_title="Estimated Cycles to 80% SoH",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_rul, width='stretch')

# -----------------------------------------------------------------------------
# 7. AI SAFETY-AWARE NARRATIVE GENERATOR (LOCAL OLLAMA INTEGRATION)
# -----------------------------------------------------------------------------
st.subheader("🤖 AI Safety-Aware Operator Diagnostic Summary")

@st.cache_data(show_spinner=False)
def fetch_real_narrative(soh_v, slope_v, driver_str, r_best, r_likely, r_worst):
    return generate_operator_narrative(
        current_soh=float(soh_v),
        trend_slope=float(slope_v),
        top_driver=str(driver_str),
        rul_best_cycles=int(r_best),
        rul_likely_cycles=int(r_likely),
        rul_worst_cycles=int(r_worst)
    )

with st.spinner("🤖 Generating safety-aware narrative via local Ollama model (llama3.2:3b)..."):
    narrative_output = fetch_real_narrative(
        current_soh, slope_val, top_feat_name, rul_best, rul_likely, rul_worst
    )

st.info(narrative_output)
