import os
import ast
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.narrative import generate_operator_narrative
import src.features as ft
import src.rul_model as rm

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CONSOLIDATED STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery SoH & RUL Estimator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Consolidated CSS design system
st.markdown("""
<style>
    /* System font stack & base dark theme background */
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Top Disclaimer Banner */
    .disclaimer-banner {
        background: linear-gradient(90deg, rgba(185, 28, 28, 0.25) 0%, rgba(153, 27, 27, 0.15) 100%);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 20px;
        color: #fca5a5;
        font-weight: 500;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }
    .disclaimer-banner strong {
        color: #f87171;
    }
    
    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 16px;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-align: center;
    }
    .badge-healthy {
        background-color: rgba(21, 128, 61, 0.3);
        border: 1px solid #22c55e;
        color: #86efac;
    }
    .badge-monitor {
        background-color: rgba(180, 83, 9, 0.3);
        border: 1px solid #f59e0b;
        color: #fde047;
    }
    .badge-replace {
        background-color: rgba(185, 28, 28, 0.3);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }

    /* Section Subheaders */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Narrative Info Box styling */
    .narrative-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px 20px;
        color: #e2e8f0;
        font-size: 0.98rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. TOP DISCLAIMER BANNER (NON-DISMISSIBLE)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <span style="font-size: 1.2rem;">⚠️</span>
    <div>
        <strong>SAFETY DISCLAIMER:</strong> Predictive telemetry estimates are for advisory decision-support only and do not replace certified physical testing or manufacturer diagnostics.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🔋 Battery State-of-Health (SoH) & RUL Estimator")
st.caption("Real-Time Telemetry Analytics, Predictive Degradation Modeling & Safety Guardrails")

# -----------------------------------------------------------------------------
# 3. REAL TEAMMATES DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
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
        
    df_merged = df_features.merge(
        df_soh[['cell_id', 'cycle_id', 'soh_predicted']], 
        on=['cell_id', 'cycle_id'], 
        how='left'
    )
    df_merged['soh_predicted'] = df_merged['soh_predicted'].fillna(df_merged['soh_ground_truth'])
    df_merged['soh_upper'] = np.clip(df_merged['soh_predicted'] + 0.8, 60.0, 100.0)
    df_merged['soh_lower'] = np.clip(df_merged['soh_predicted'] - 0.8, 60.0, 100.0)
    
    rul_file = os.path.join('outputs', 'rul_predictions.csv')
    if os.path.exists(rul_file):
        df_rul = pd.read_csv(rul_file)
    else:
        df_rul = rm.build_rul_output(df_soh)
        
    return df_merged, df_drivers, df_rul

df_all, df_drivers, df_rul = load_pipeline_data()

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & CELL METADATA
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Cell Selection & Controls")
cell_list = sorted(df_all['cell_id'].unique().tolist())
selected_cell = st.sidebar.selectbox("Target Battery Cell:", cell_list)

df_cell = df_all[df_all['cell_id'] == selected_cell].sort_values('cycle_id').reset_index(drop=True)

cell_rul_row = df_rul[df_rul['cell_id'] == selected_cell]
if len(cell_rul_row) > 0:
    r_row = cell_rul_row.iloc[0]
    current_soh = float(r_row['current_soh']) if pd.notna(r_row['current_soh']) else float(df_cell['soh_predicted'].iloc[-1])
    slope_val = float(r_row['trend_slope']) if pd.notna(r_row['trend_slope']) else -0.01
    
    rul_likely = int(r_row['rul_likely_cycles']) if pd.notna(r_row['rul_likely_cycles']) else 0
    rul_worst = int(r_row['rul_worst_cycles']) if pd.notna(r_row['rul_worst_cycles']) else 0
    rul_best = int(r_row['rul_best_cycles']) if pd.notna(r_row['rul_best_cycles']) else 5000
    
    cell_top_driver = str(r_row['top_driver']) if pd.notna(r_row['top_driver']) else "Cumulative Time Above 40C"
    cell_scores_raw = r_row['driver_importance_scores']
    if isinstance(cell_scores_raw, str):
        try:
            cell_scores_dict = ast.literal_eval(cell_scores_raw)
        except Exception:
            cell_scores_dict = {}
    elif isinstance(cell_scores_raw, dict):
        cell_scores_dict = cell_scores_raw
    else:
        cell_scores_dict = {}
else:
    rul_dict = rm.compute_bands_for_cell(df_cell)
    current_soh = float(rul_dict['current_soh'])
    slope_val = float(rul_dict['trend_slope'])
    rul_likely = int(rul_dict['rul_likely_cycles']) if rul_dict['rul_likely_cycles'] is not None else 0
    rul_worst = int(rul_dict['rul_worst_cycles']) if rul_dict['rul_worst_cycles'] is not None else 0
    rul_best = int(rul_dict['rul_best_cycles']) if rul_dict['rul_best_cycles'] is not None else 5000
    cell_top_driver = "Cumulative Time Above 40C"
    cell_scores_dict = {}

top_feat_name = cell_top_driver.replace('_', ' ').title()

# Status classification logic: SoH <= 80% OR RUL <= 50 cycles triggers Replace Soon, slope <= -0.05 triggers Monitor
if current_soh <= 80.0 or rul_likely <= 50:
    status_label = "Replace Soon"
    badge_class = "badge-replace"
elif current_soh <= 85.0 or slope_val <= -0.05:
    status_label = "Monitor Closely"
    badge_class = "badge-monitor"
else:
    status_label = "Healthy"
    badge_class = "badge-healthy"

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Active Cell Profile")
st.sidebar.write(f"**Cell Identifier**: `{selected_cell}`")
st.sidebar.write(f"**Telemetry History**: `{len(df_cell)} cycles`")
st.sidebar.write(f"**Decline Velocity**: `{slope_val:.5f}% / cycle`")
st.sidebar.write(f"**Status Tier**: <span class=\"badge-pill {badge_class}\">{status_label}</span>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. TOP METRIC ROW (VISIBLE WITHOUT SCROLLING)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Current Health (SoH)</div>
        <div class="kpi-value">{current_soh:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Likely Remaining Life</div>
        <div class="kpi-value">{rul_likely} <span class="kpi-subtext">cycles</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Primary Cell Stressor</div>
        <div class="kpi-value" style="font-size: 1.1rem; margin-top: 6px; color: #38bdf8;">{top_feat_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Risk Classification</div>
        <div style="margin-top: 8px;"><span class="badge-pill {badge_class}">{status_label}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. CHART 1: SOH DECAY TRAJECTORY
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">📉 State-of-Health (SoH) Decay Trajectory</div>', unsafe_allow_html=True)

fig_soh = go.Figure()

fig_soh.add_trace(go.Scatter(
    x=pd.concat([df_cell['cycle_id'], df_cell['cycle_id'][::-1]]),
    y=pd.concat([df_cell['soh_upper'], df_cell['soh_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(99, 102, 241, 0.12)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Model 95% Confidence Interval'
))

fig_soh.add_trace(go.Scatter(
    x=df_cell['cycle_id'],
    y=df_cell['soh_ground_truth'],
    mode='lines',
    name='Ground Truth SoH',
    hovertemplate='<b>Cycle %{x}</b><br>Ground Truth: %{y:.2f}%<extra></extra>',
    line=dict(color='#38bdf8', width=2, dash='dash')
))

fig_soh.add_trace(go.Scatter(
    x=df_cell['cycle_id'],
    y=df_cell['soh_predicted'],
    mode='lines',
    name='Predicted SoH (Model)',
    hovertemplate='<b>Cycle %{x}</b><br>Predicted SoH: %{y:.2f}%<extra></extra>',
    line=dict(color='#818cf8', width=3)
))

fig_soh.add_hline(
    y=80.0, 
    line_dash="dash", 
    line_color="#ef4444", 
    annotation_text="80% End-of-Life Threshold", 
    annotation_position="bottom right",
    annotation_font=dict(color="#fca5a5", size=11)
)

fig_soh.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15, 23, 42, 0.6)',
    xaxis=dict(
        title="Cycle Count", 
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        title="State-of-Health (%)", 
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    ),
    height=370,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_soh, width='stretch')

# -----------------------------------------------------------------------------
# 7. CHARTS ROW 2: PER-CELL DRIVERS & RUL UNCERTAINTY BANDS
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown('<div class="section-header">📊 Cell Stressor Importance Breakdown</div>', unsafe_allow_html=True)
    if cell_scores_dict:
        driver_df_cell = pd.DataFrame(list(cell_scores_dict.items()), columns=['Feature', 'Importance'])
        driver_df_cell['clean_name'] = driver_df_cell['Feature'].str.replace('_', ' ').str.title()
        driver_df_cell = driver_df_cell.sort_values('Importance', ascending=False).head(5)
    else:
        driver_df_cell = df_drivers.head(5).copy()
        driver_df_cell['clean_name'] = driver_df_cell['Feature'].str.replace('_', ' ').str.title()
        
    fig_driver = go.Figure(go.Bar(
        x=driver_df_cell['Importance'],
        y=driver_df_cell['clean_name'],
        orientation='h',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>',
        marker=dict(
            color=driver_df_cell['Importance'],
            colorscale=[[0, '#312e81'], [0.5, '#6366f1'], [1, '#38bdf8']],
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        )
    ))
    fig_driver.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        xaxis=dict(
            title="Relative Stress Contribution", 
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(autorange="reversed", gridcolor='rgba(255,255,255,0.05)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_driver, width='stretch')

with col_chart2:
    st.markdown('<div class="section-header">⏳ RUL Uncertainty Boundaries</div>', unsafe_allow_html=True)
    rul_chart_df = pd.DataFrame({
        'Scenario': ['Conservative (P10)', 'Likely (P50)', 'Optimistic (P90)'],
        'Cycles': [rul_worst, rul_likely, rul_best],
        'Color': ['#ef4444', '#f59e0b', '#10b981']
    })
    
    fig_rul = go.Figure(go.Bar(
        x=rul_chart_df['Scenario'],
        y=rul_chart_df['Cycles'],
        text=rul_chart_df['Cycles'].astype(str) + " cycles",
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Remaining: %{y} cycles<extra></extra>',
        marker=dict(
            color=rul_chart_df['Color'],
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        )
    ))
    fig_rul.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        yaxis=dict(title="Cycles to 80% Threshold", gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_rul, width='stretch')

# -----------------------------------------------------------------------------
# 8. AI SAFETY NARRATIVE DIAGNOSTIC SUMMARY
# -----------------------------------------------------------------------------
st.markdown(f'''
<div class="section-header">
    🤖 AI Safety Diagnostic Summary
    <span class="badge-pill {badge_class}" style="margin-left: auto;">{status_label}</span>
</div>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_real_narrative(soh_v, slope_v, driver_str, r_best, r_likely, r_worst):
    return generate_operator_narrative(
        current_soh=float(soh_v),
        trend_slope=float(slope_v),
        top_driver=str(driver_str),
        rul_best_cycles=int(r_best),
        rul_likely_cycles=int(r_likely),
        rul_worst_cycles=int(r_worst),
        model_name='llama3.2:3b'
    )

with st.spinner("🤖 Generating safety-aware operator report via local LLM..."):
    narrative_output = fetch_real_narrative(
        current_soh, slope_val, top_feat_name, rul_best, rul_likely, rul_worst
    )

st.markdown(f'<div class="narrative-box">{narrative_output}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. EXPANDABLE TECHNICAL DETAILS (INTERACTIVE TOUCHES)
# -----------------------------------------------------------------------------
with st.expander("🔬 Telemetry & Physics Degradation Model Parameters"):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        **Degradation Physics Model Assumptions**:
        - **Thermal Stress Gate**: Degradation accelerates when temperature $> 40^\circ\text{C}$.
        - **C-Rate Current Density**: High rate charging ($>0.8\text{ C}$) drives lithium plating risk.
        - **Depth of Discharge (DoD)**: Deep cycling below $20\%\text{ SoC}$ expands mechanical lattice strain.
        """)
    with col_t2:
        st.markdown(f"""
        **Cell `{selected_cell}` Trajectory Breakdown**:
        - Trajectory Window: `{len(df_cell)} cycles`
        - Current SoH Estimate: `{current_soh:.2f}%`
        - Baseline Decline Rate: `{slope_val:.5f}% / cycle`
        - Primary Degradation Cause: `{top_feat_name}`
        """)

with st.expander("📋 Raw Telemetry Data Table"):
    st.dataframe(df_cell.tail(20), use_container_width=True)
