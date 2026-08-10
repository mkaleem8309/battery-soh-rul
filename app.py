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
# 1. PAGE CONFIG & CONSOLIDATED MODERN DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery SoH & RUL Estimator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Consolidated CSS design system (Modern UI patterns, glassmorphism, entrance animations)
st.markdown("""
<style>
    /* System font stack & base dark theme background */
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #080c14;
        color: #f8fafc;
    }
    
    /* Smooth CSS Entrance Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .main-animated-container {
        animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1);
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
    
    /* Top Header Energy Banner Graphic */
    .header-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%),
                    radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.12) 0%, transparent 40%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* Cell Selector Card Grid / Carousel styling */
    .carousel-container {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 6px 2px 14px 2px;
        scrollbar-width: thin;
        scrollbar-color: #334155 #0f172a;
    }
    .cell-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        min-width: 130px;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
    }
    .cell-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
    }
    .cell-card-active {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border: 2px solid #38bdf8 !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.15);
    }
    .kpi-header-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin-bottom: 8px;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.9rem;
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
        transition: all 0.25s ease;
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
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 18px 22px;
        color: #e2e8f0;
        font-size: 0.98rem;
        line-height: 1.65;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
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

# Top Header Energy Banner Graphic
st.markdown("""
<div class="header-banner">
    <div class="header-title">🔋 Battery State-of-Health (SoH) & RUL Estimator</div>
    <div class="header-subtitle">Real-Time Telemetry Analytics, Predictive Degradation Modeling & Safety Guardrails</div>
</div>
""", unsafe_allow_html=True)

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
# 4. INTERACTIVE CELL SELECTOR CAROUSEL & SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
cell_list = sorted(df_all['cell_id'].unique().tolist())

# Sync selected cell via session_state
if "selected_cell" not in st.session_state:
    st.session_state["selected_cell"] = cell_list[0]

st.markdown('<div class="section-header">📱 Interactive Battery Cell Selector</div>', unsafe_allow_html=True)

# Render interactive cell cards in a clean horizontal grid (carousel cards)
card_cols = st.columns(min(len(cell_list), 6))
for i, c_id in enumerate(cell_list):
    col_idx = i % 6
    if i > 0 and col_idx == 0:
        # Wrap into next visual row if more than 6 cells
        card_cols = st.columns(min(len(cell_list) - i, 6))
        
    c_rul_info = df_rul[df_rul['cell_id'] == c_id]
    if len(c_rul_info) > 0:
        c_soh = float(c_rul_info.iloc[0]['current_soh'])
        c_rul_cycles = int(c_rul_info.iloc[0]['rul_likely_cycles'])
        c_slope = float(c_rul_info.iloc[0]['trend_slope'])
    else:
        c_soh = 90.0
        c_rul_cycles = 500
        c_slope = -0.01

    if c_soh <= 80.0 or c_rul_cycles <= 50:
        c_status = "Replace"
        c_color = "#ef4444"
        c_icon = "🔴"
    elif c_soh <= 85.0 or c_slope <= -0.05:
        c_status = "Monitor"
        c_color = "#f59e0b"
        c_icon = "🟡"
    else:
        c_status = "Healthy"
        c_color = "#22c55e"
        c_icon = "🟢"

    is_selected = (c_id == st.session_state["selected_cell"])
    btn_label = f"{c_icon} {c_id}\n{c_soh:.1f}% ({c_status})"
    
    with card_cols[col_idx]:
        if st.button(
            btn_label,
            key=f"carousel_btn_{c_id}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            st.session_state["selected_cell"] = c_id
            st.rerun()

st.sidebar.header("🕹️ Cell Controls & Sidebar Sync")
sidebar_selected = st.sidebar.selectbox(
    "Target Battery Cell (Sidebar Sync):", 
    cell_list, 
    index=cell_list.index(st.session_state["selected_cell"]) if st.session_state["selected_cell"] in cell_list else 0
)

# Synchronize if changed in sidebar
if sidebar_selected != st.session_state["selected_cell"]:
    st.session_state["selected_cell"] = sidebar_selected
    st.rerun()

selected_cell = st.session_state["selected_cell"]
df_cell = df_all[df_all['cell_id'] == selected_cell].sort_values('cycle_id').reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5. METADATA & CONTINUOUS SOH GRADIENT CALCULATIONS
# -----------------------------------------------------------------------------
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

# Continuous SoH Color Gradient calculation (Emerald -> Amber -> Red)
soh_norm = np.clip((current_soh - 80.0) / 20.0, 0.0, 1.0)
if soh_norm >= 0.5:
    # 90% to 100% (Amber to Emerald)
    factor = (soh_norm - 0.5) * 2.0
    r_val = int(245 * (1 - factor) + 34 * factor)
    g_val = int(158 * (1 - factor) + 197 * factor)
    b_val = int(11 * (1 - factor) + 94 * factor)
else:
    # 80% to 90% (Red to Amber)
    factor = soh_norm * 2.0
    r_val = int(239 * (1 - factor) + 245 * factor)
    g_val = int(68 * (1 - factor) + 158 * factor)
    b_val = int(68 * (1 - factor) + 11 * factor)

soh_gradient_color = f"rgb({r_val}, {g_val}, {b_val})"

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Active Cell Profile")
st.sidebar.write(f"**Cell Identifier**: `{selected_cell}`")
st.sidebar.write(f"**Telemetry History**: `{len(df_cell)} cycles`")
st.sidebar.write(f"**Decline Velocity**: `{slope_val:.5f}% / cycle`")
st.sidebar.write(f"**Status Tier**: <span class=\"badge-pill {badge_class}\">{status_label}</span>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. MAIN ANIMATED CONTAINER & KPI SUMMARY ROW WITH LOCAL SVG ICONS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-animated-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

# SVG Icons (0 network dependencies)
svg_battery = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><rect x="2" y="7" width="16" height="10" rx="2"/><path d="M22 11v2"/></svg>"""
svg_clock = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""
svg_thermo = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>"""
svg_shield = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header-row">{svg_battery} <span class="kpi-title">Current Health (SoH)</span></div>
        <div class="kpi-value" style="color: {soh_gradient_color};">{current_soh:.1f}%</div>
        <div style="width: 100%; background: rgba(255,255,255,0.1); height: 6px; border-radius: 4px; margin-top: 10px; overflow: hidden;">
            <div style="width: {current_soh}%; background: {soh_gradient_color}; height: 100%; border-radius: 4px; transition: width 0.4s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header-row">{svg_clock} <span class="kpi-title">Likely Remaining Life</span></div>
        <div class="kpi-value">{rul_likely} <span class="kpi-subtext">cycles</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header-row">{svg_thermo} <span class="kpi-title">Primary Cell Stressor</span></div>
        <div class="kpi-value" style="font-size: 1.05rem; margin-top: 6px; color: #38bdf8;">{top_feat_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header-row">{svg_shield} <span class="kpi-title">Risk Classification</span></div>
        <div style="margin-top: 8px;"><span class="badge-pill {badge_class}">{status_label}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. CHART 1: SOH DECAY TRAJECTORY
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
# 8. CHARTS ROW 2: PER-CELL DRIVERS & RUL UNCERTAINTY BANDS
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
# 9. AI SAFETY NARRATIVE DIAGNOSTIC SUMMARY
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
# 10. EXPANDABLE TECHNICAL DETAILS & PAGINATED TELEMETRY TABLE
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

with st.expander("📋 Paginated Telemetry Data Table"):
    PAGE_SIZE = 6
    total_rows = len(df_cell)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    
    if "table_page" not in st.session_state:
        st.session_state["table_page"] = 0
        
    # Clamp table_page
    st.session_state["table_page"] = max(0, min(st.session_state["table_page"], total_pages - 1))
    current_page = st.session_state["table_page"]
    
    start_idx = current_page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_rows)
    
    st.dataframe(df_cell.iloc[start_idx:end_idx], use_container_width=True)
    
    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
    with p_col1:
        if st.button("⬅️ Previous Page", disabled=(current_page == 0), use_container_width=True):
            st.session_state["table_page"] -= 1
            st.rerun()
    with p_col2:
        st.markdown(f"<div style='text-align: center; color: #94a3b8; padding-top: 6px;'>Page <strong>{current_page + 1}</strong> of <strong>{total_pages}</strong> ({total_rows} total cycles)</div>", unsafe_allow_html=True)
    with p_col3:
        if st.button("Next Page ➡️", disabled=(current_page == total_pages - 1), use_container_width=True):
            st.session_state["table_page"] += 1
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # Close main-animated-container
