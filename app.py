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
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Consolidated CSS design system (Step 1 Typography & Step 2 Dark Fintech Color Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

    /* Global typography & fintech dark theme background (#0a0e14) */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp {
        background-color: #0a0e14;
        color: #f8fafc;
    }
    
    /* Smooth CSS Entrance Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .main-animated-container {
        animation: fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    /* Top Disclaimer Banner (Softened Hairline Style) */
    .disclaimer-banner {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #ef4444;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.90rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .disclaimer-banner strong {
        color: #ef4444;
    }
    
    /* Top Header Energy Banner Graphic */
    .header-banner {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        font-weight: 500;
        letter-spacing: -0.01em;
    }

    /* Pill-Style Toggles & Selector Styling */
    .pill-group {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 30px;
        padding: 4px;
        display: inline-flex;
        gap: 4px;
        margin-bottom: 18px;
    }
    .pill-option {
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .pill-active {
        background: #3b82f6;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
    }

    /* Status-Specific Active Carousel Button CSS Overrides */
    .cell-pill-active-healthy button {
        background-color: rgba(21, 128, 61, 0.25) !important;
        border: 1px solid #22c55e !important;
        color: #86efac !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.3) !important;
    }
    .cell-pill-active-monitor button {
        background-color: rgba(180, 83, 9, 0.25) !important;
        border: 1px solid #f59e0b !important;
        color: #fde047 !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.3) !important;
    }
    .cell-pill-active-replace button {
        background-color: rgba(185, 28, 28, 0.25) !important;
        border: 1px solid #ef4444 !important;
        color: #fca5a5 !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.3) !important;
    }

    /* Cell Selector Card Grid / Carousel styling (Rounded Pill Cards) */
    .carousel-container {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 6px 2px 14px 2px;
        scrollbar-width: thin;
        scrollbar-color: #334155 #0a0e14;
    }
    /* NOTE: cell selection now uses st.button with data-testid targeting
       (see stBaseButton-primary/secondary rules below), not these classes.
       Kept only for any HTML-rendered cards elsewhere in the file. */
    .cell-card {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 10px 16px;
        min-width: 130px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .cell-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .cell-card-active {
        background: #181d28;
        border: 1px solid #3b82f6 !important;
    }

    /* KPI Metric Cards (Restrained Depth, Hairline Borders, Electric Blue Accent #3b82f6) */
    .kpi-card {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        transition: border-color 0.2s ease, transform 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .kpi-header-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin-bottom: 8px;
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .kpi-value {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #3b82f6;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Functionally Meaningful Status Badges */
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
        background-color: rgba(21, 128, 61, 0.2);
        border: 1px solid #22c55e;
        color: #86efac;
    }
    .badge-monitor {
        background-color: rgba(180, 83, 9, 0.2);
        border: 1px solid #f59e0b;
        color: #fde047;
    }
    .badge-replace {
        background-color: rgba(185, 28, 28, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }

    /* Section Subheaders */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #f8fafc;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Narrative Info Box styling */
    .narrative-box {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 20px 24px;
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.65;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-align: center;
        transition: all 0.25s ease;
    }

    /* Step 5 Custom Animated AI Thinking Pulse Orb Loading State */
    @keyframes aiOrbGlow {
        0%, 100% {
            transform: scale(0.94);
            box-shadow:
                0 0 20px 4px rgba(59, 130, 246, 0.45),
                0 0 40px 12px rgba(59, 130, 246, 0.20);
        }
        50% {
            transform: scale(1.06);
            box-shadow:
                0 0 32px 8px rgba(59, 130, 246, 0.65),
                0 0 60px 20px rgba(59, 130, 246, 0.30);
        }
    }
    @keyframes aiOrbSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes aiParticleTwinkle {
        0%, 100% { opacity: 0.15; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.15); }
    }
    .ai-thinking-container {
        background: #12151c;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 28px 24px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
    }
    .ai-orb-wrap {
        position: relative;
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ai-thinking-orb {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #93c5fd 0%, #3b82f6 45%, #1e40af 100%);
        animation: aiOrbGlow 1.8s ease-in-out infinite;
        z-index: 2;
    }
    .ai-orb-ring {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50%;
        border: 1px dashed rgba(96, 165, 250, 0.35);
        animation: aiOrbSpin 6s linear infinite;
        z-index: 1;
    }
    .ai-particle {
        position: absolute;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: #93c5fd;
        animation: aiParticleTwinkle 1.6s ease-in-out infinite;
    }
    .ai-particle:nth-child(1) { top: -2px;  left: 8px;  animation-delay: 0s; }
    .ai-particle:nth-child(2) { top: 10px;  right: -4px; animation-delay: 0.3s; }
    .ai-particle:nth-child(3) { bottom: 2px; left: -4px; animation-delay: 0.6s; }
    .ai-particle:nth-child(4) { bottom: -2px; right: 10px; animation-delay: 0.9s; }
    .ai-thinking-text {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 500;
        letter-spacing: -0.01em;
        line-height: 1.5;
    }
    .ai-thinking-text strong {
        color: #f8fafc;
        font-weight: 700;
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

    /* Welcome / hero screen — Stellar-inspired dark hero, own design system */
    .hero-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 78vh;
        padding: 2rem 1rem;
        position: relative;
        overflow: hidden;
    }
    .hero-glow {
        position: absolute;
        top: -10%;
        left: 50%;
        transform: translateX(-50%);
        width: 640px;
        height: 640px;
        background: radial-gradient(circle, rgba(59,130,246,0.22) 0%, rgba(59,130,246,0.06) 40%, transparent 70%);
        filter: blur(10px);
        pointer-events: none;
        z-index: 0;
    }
    @keyframes heroParticleFloat {
        0%   { opacity: 0.15; transform: translate(0, 0) scale(0.8); }
        50%  { opacity: 0.9;  transform: translate(var(--drift-x, 8px), var(--drift-y, -14px)) scale(1.2); }
        100% { opacity: 0.15; transform: translate(0, 0) scale(0.8); }
    }
    .hero-particle {
        position: absolute;
        border-radius: 50%;
        background: #93c5fd;
        pointer-events: none;
        z-index: 0;
        animation: heroParticleFloat ease-in-out infinite;
    }
    .hero-particle:nth-child(2) { width: 5px; height: 5px; top: 12%; left: 28%; --drift-x: 10px; --drift-y: -18px; animation-duration: 5.5s; animation-delay: 0s; background: #93c5fd; }
    .hero-particle:nth-child(3) { width: 3px; height: 3px; top: 22%; left: 68%; --drift-x: -14px; --drift-y: -10px; animation-duration: 6.8s; animation-delay: 0.7s; background: #60a5fa; }
    .hero-particle:nth-child(4) { width: 4px; height: 4px; top: 55%; left: 18%; --drift-x: 12px; --drift-y: 14px; animation-duration: 7.4s; animation-delay: 1.4s; background: #34d399; }
    .hero-particle:nth-child(5) { width: 3px; height: 3px; top: 8%; left: 52%; --drift-x: -8px; --drift-y: 16px; animation-duration: 5.9s; animation-delay: 2.1s; background: #93c5fd; }
    .hero-particle:nth-child(6) { width: 5px; height: 5px; top: 62%; left: 74%; --drift-x: -12px; --drift-y: -12px; animation-duration: 6.3s; animation-delay: 0.4s; background: #60a5fa; }
    .hero-particle:nth-child(7) { width: 3px; height: 3px; top: 35%; left: 8%; --drift-x: 16px; --drift-y: -8px; animation-duration: 7.9s; animation-delay: 1.8s; background: #93c5fd; }
    .hero-particle:nth-child(8) { width: 4px; height: 4px; top: 40%; left: 88%; --drift-x: -10px; --drift-y: 12px; animation-duration: 6.6s; animation-delay: 1.1s; background: #34d399; }
    .hero-badge {
        position: relative;
        z-index: 1;
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border: 1px solid rgba(59,130,246,0.35);
        background: rgba(59,130,246,0.08);
        border-radius: 999px;
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        line-height: 1.15;
        color: #f8fafc;
        margin-bottom: 1rem;
        max-width: 780px;
    }
    .hero-title .accent {
        background: linear-gradient(90deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero-subtitle {
        position: relative;
        z-index: 1;
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 620px;
        line-height: 1.7;
        margin-bottom: 2.2rem;
    }
    .hero-stats {
        position: relative;
        z-index: 1;
        display: flex;
        gap: 2.2rem;
        margin-bottom: 2.4rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    .hero-stat-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.6rem;
        color: #f8fafc;
    }
    .hero-stat-label {
        font-size: 0.78rem;
        color: #64748b;
        letter-spacing: 0.03em;
    }
    div[data-testid="stButton"] button[kind="primary"].hero-cta,
    .hero-cta-wrap button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        border: none !important;
        padding: 0.75rem 2.4rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(59,130,246,0.35) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .hero-cta-wrap button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(59,130,246,0.45) !important;
    }
    .hero-footnote {
        position: relative;
        z-index: 1;
        margin-top: 1.6rem;
        font-size: 0.75rem;
        color: #475569;
        max-width: 520px;
    }

    /* Fleet Overview page — dark management-dashboard style */
    .ov-header {
        margin-bottom: 1.6rem;
    }
    .ov-eyebrow {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.3rem;
    }
    .ov-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        color: #f8fafc;
    }
    .ov-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }
    .ov-kpi-card {
        background: #12151c;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1.4rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ov-kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59,130,246,0.4);
    }
    .ov-kpi-label {
        font-size: 0.78rem;
        color: #64748b;
        letter-spacing: 0.02em;
        margin-bottom: 0.4rem;
    }
    .ov-kpi-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
    }
    .ov-cell-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .ov-cell-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }
    .ov-cell-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .ov-cell-id {
        font-weight: 700;
        color: #f8fafc;
        font-size: 0.95rem;
    }
    .ov-cell-badge {
        font-size: 0.7rem;
        font-weight: 600;
        border: 1px solid;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        letter-spacing: 0.02em;
    }
    .ov-cell-soh {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .ov-cell-meta {
        font-size: 0.78rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1a. SHARED DATA LOADER (moved above page routing so both Overview and
#     Detail pages can use it)
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

    # Restore temperature_max from raw telemetry dataset if not already in features
    synthetic_file = os.path.join('data', 'synthetic_battery_data.csv')
    if os.path.exists(synthetic_file) and 'temperature_max' not in df_features.columns:
        try:
            raw_df = pd.read_csv(synthetic_file)
            if 'temperature_max' in raw_df.columns:
                df_features = df_features.merge(
                    raw_df[['cell_id', 'cycle_id', 'temperature_max']],
                    on=['cell_id', 'cycle_id'],
                    how='left'
                )
        except Exception:
            pass

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


def classify_tier(soh, rul_likely, slope):
    if soh <= 80.0 or rul_likely <= 50:
        return "Critical", "#ef4444", "rgba(239, 68, 68, 0.14)"
    elif soh <= 85.0 or slope <= -0.05:
        return "Monitor", "#f59e0b", "rgba(245, 158, 11, 0.14)"
    else:
        return "Healthy", "#22c55e", "rgba(34, 197, 94, 0.14)"


# -----------------------------------------------------------------------------
# 1b. PAGE ROUTER — welcome -> fleet overview -> cell detail
# -----------------------------------------------------------------------------
if "app_page" not in st.session_state:
    st.session_state["app_page"] = "welcome"

# ---- PAGE 1: WELCOME / HERO SCREEN -----------------------------------------
if st.session_state["app_page"] == "welcome":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-glow"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-particle"></div>
        <div class="hero-badge">Predictive Battery Analytics</div>
        <div class="hero-title">
            Battery State-of-Health &amp;<br><span class="accent">Remaining Useful Life</span> Estimator
        </div>
        <div class="hero-subtitle">
            Turns raw charge/discharge telemetry into SoH predictions, uncertainty-banded
            RUL forecasts, and plain-language operator guidance — so degradation shows up
            weeks before it becomes a failure.
        </div>
        <div class="hero-stats">
            <div><div class="hero-stat-value">18</div><div class="hero-stat-label">CELLS MONITORED</div></div>
            <div><div class="hero-stat-value">98.6%</div><div class="hero-stat-label">MODEL R²</div></div>
            <div><div class="hero-stat-value">3</div><div class="hero-stat-label">RUL BANDS</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, _mid, _ = st.columns([1, 1, 1])
    with _mid:
        st.markdown('<div class="hero-cta-wrap">', unsafe_allow_html=True)
        if st.button("Launch Dashboard", type="primary", use_container_width=True, key="hero_launch_btn"):
            st.session_state["app_page"] = "overview"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; justify-content:center;">
        <div class="hero-footnote">
            This is a data-driven estimate, not a substitute for manufacturer testing or
            certified diagnostics. Do not use as the sole basis for safety-critical decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ---- PAGE 2: FLEET OVERVIEW ------------------------------------------------
if st.session_state["app_page"] == "overview":
    df_all_ov, df_drivers_ov, df_rul_ov = load_pipeline_data()

    tiers = df_rul_ov.apply(
        lambda r: classify_tier(r["current_soh"], r["rul_likely_cycles"], r["trend_slope"])[0], axis=1
    )
    n_healthy = (tiers == "Healthy").sum()
    n_monitor = (tiers == "Monitor").sum()
    n_critical = (tiers == "Critical").sum()
    fleet_avg_soh = df_rul_ov["current_soh"].mean()

    st.markdown("""
    <div class="ov-header">
        <div>
            <div class="ov-eyebrow">Fleet Overview</div>
            <div class="ov-title">Battery Fleet Management</div>
            <div class="ov-subtitle">Live health status across every monitored cell</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, color in [
        (k1, "Fleet Average SoH", f"{fleet_avg_soh:.1f}%", "#3b82f6"),
        (k2, "Healthy", str(n_healthy), "#22c55e"),
        (k3, "Monitor Closely", str(n_monitor), "#f59e0b"),
        (k4, "Replace Soon", str(n_critical), "#ef4444"),
    ]:
        with col:
            st.markdown(f"""
            <div class="ov-kpi-card">
                <div class="ov-kpi-label">{label}</div>
                <div class="ov-kpi-value" style="color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">- Cell Fleet Grid</div>', unsafe_allow_html=True)

    grid_cols = st.columns(3)
    for idx, row in df_rul_ov.sort_values("cell_id").reset_index(drop=True).iterrows():
        tier_label, tier_color, tier_bg = classify_tier(row["current_soh"], row["rul_likely_cycles"], row["trend_slope"])
        col = grid_cols[idx % 3]
        with col:
            st.markdown(f"""
            <div class="ov-cell-card" style="border-color:{tier_color}44; background:{tier_bg}; padding-bottom:0.4rem;">
                <div class="ov-cell-top">
                    <span class="ov-cell-id">{row['cell_id']}</span>
                    <span class="ov-cell-badge" style="color:{tier_color}; border-color:{tier_color}66;">{tier_label}</span>
                </div>
                <div class="ov-cell-soh">{row['current_soh']:.1f}<span style="font-size:0.85rem; color:#64748b;">% SoH</span></div>
            """, unsafe_allow_html=True)

            cell_trend = df_all_ov[df_all_ov["cell_id"] == row["cell_id"]].sort_values("cycle_id")
            if len(cell_trend) > 1:
                _r, _g, _b = int(tier_color[1:3], 16), int(tier_color[3:5], 16), int(tier_color[5:7], 16)
                _y_min = cell_trend["soh_predicted"].min()
                _y_max = cell_trend["soh_predicted"].max()
                _y_span = max(_y_max - _y_min, 0.5)  # avoid a zero-height range on a near-flat cell
                _y_pad = _y_span * 0.15
                spark = go.Figure()
                # baseline trace added FIRST, at the series' own minimum (not 0), so the
                # fill below doesn't pull the y-axis down to 0 and flatten the visible slope
                spark.add_trace(go.Scatter(
                    x=cell_trend["cycle_id"],
                    y=[_y_min - _y_pad] * len(cell_trend),
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                spark.add_trace(go.Scatter(
                    x=cell_trend["cycle_id"],
                    y=cell_trend["soh_predicted"],
                    mode="lines",
                    line=dict(color=tier_color, width=2.5),
                    fill="tonexty",
                    fillcolor=f"rgba({_r},{_g},{_b},0.14)",
                    hoverinfo="skip",
                    showlegend=False,
                ))
                spark.update_layout(
                    height=80,
                    margin=dict(l=0, r=0, t=4, b=4),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, range=[_y_min - _y_pad, _y_max + _y_pad]),
                    showlegend=False,
                )
                st.plotly_chart(
                    spark, use_container_width=True, config={"displayModeBar": False},
                    key=f"ov_spark_{row['cell_id']}"
                )

            st.markdown(f"""
                <div class="ov-cell-meta">RUL (likely): {int(row['rul_likely_cycles'])} cycles</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("View Details", key=f"ov_view_{row['cell_id']}", use_container_width=True):
                st.session_state["selected_cell"] = row["cell_id"]
                st.session_state["app_page"] = "detail"
                st.rerun()

    st.stop()

# ---- PAGE 3: CELL DETAIL (existing per-cell dashboard) ---------------------
if st.button("- Back to Fleet Overview", key="back_to_overview_btn"):
    st.session_state["app_page"] = "overview"
    st.rerun()

# -----------------------------------------------------------------------------
# 2. TOP DISCLAIMER BANNER (NON-DISMISSIBLE)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <span style="font-weight: 700;">NOTICE</span>
    <div>
        <strong>SAFETY DISCLAIMER:</strong> Predictive telemetry estimates are for advisory decision-support only and do not replace certified physical testing or manufacturer diagnostics.
    </div>
</div>
""", unsafe_allow_html=True)

# Top Header Energy Banner Graphic
st.markdown("""
<div class="header-banner">
    <div class="header-title">Battery State-of-Health (SoH) & RUL Estimator</div>
    <div class="header-subtitle">Real-Time Telemetry Analytics, Predictive Degradation Modeling & Safety Guardrails</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD DATA FOR DETAIL PAGE (function defined earlier in the page router)
# -----------------------------------------------------------------------------
df_all, df_drivers, df_rul = load_pipeline_data()

# -----------------------------------------------------------------------------
# 4. INTERACTIVE CELL SELECTOR CAROUSEL & SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
cell_list = sorted(df_all['cell_id'].unique().tolist())

# Sync selected cell via session_state
if "selected_cell" not in st.session_state:
    st.session_state["selected_cell"] = cell_list[0]

active_cell_id = st.session_state["selected_cell"]
c_rul_active = df_rul[df_rul['cell_id'] == active_cell_id]
if len(c_rul_active) > 0:
    act_soh = float(c_rul_active.iloc[0]['current_soh'])
    act_rul = int(c_rul_active.iloc[0]['rul_likely_cycles'])
    act_slope = float(c_rul_active.iloc[0]['trend_slope'])
else:
    act_soh, act_rul, act_slope = 90.0, 500, -0.01

if act_soh <= 80.0 or act_rul <= 50:
    act_bg = "rgba(185, 28, 28, 0.35)"
    act_border = "#ef4444"
    act_text = "#fca5a5"
    act_glow = "rgba(239, 68, 68, 0.45)"
elif act_soh <= 85.0 or act_slope <= -0.05:
    act_bg = "rgba(180, 83, 9, 0.35)"
    act_border = "#f59e0b"
    act_text = "#fde047"
    act_glow = "rgba(245, 158, 11, 0.45)"
else:
    act_bg = "rgba(21, 128, 61, 0.35)"
    act_border = "#22c55e"
    act_text = "#86efac"
    act_glow = "rgba(34, 197, 94, 0.45)"

# Inject strict dynamic CSS rule targeting Streamlit's native stBaseButton-primary
st.markdown(f"""
<style>
    button[data-testid="stBaseButton-primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {act_bg} !important;
        border: 1.5px solid {act_border} !important;
        color: {act_text} !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px {act_glow} !important;
    }}
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] span {{
        color: {act_text} !important;
        font-weight: 700 !important;
    }}
    button[data-testid="stBaseButton-secondary"] {{
        background-color: #12151c !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #94a3b8 !important;
        transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease !important;
    }}
    button[data-testid="stBaseButton-secondary"]:hover {{
        border-color: #3b82f6 !important;
        color: #f8fafc !important;
        transform: translateY(-2px);
    }}
    button[data-testid="stBaseButton-primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        transition: box-shadow 0.25s ease, transform 0.2s ease !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">- Interactive Battery Cell Selector</div>', unsafe_allow_html=True)

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
        c_icon = "CRITICAL"
    elif c_soh <= 85.0 or c_slope <= -0.05:
        c_status = "Monitor"
        c_icon = "MONITOR"
    else:
        c_status = "Healthy"
        c_icon = "HEALTHY"

    is_selected = (c_id == st.session_state["selected_cell"])
    btn_label = f"{c_icon} {c_id} ({c_soh:.1f}%)"
            
    with card_cols[col_idx]:
        if st.button(
            btn_label,
            key=f"carousel_btn_{c_id}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            st.session_state["selected_cell"] = c_id
            st.rerun()

st.sidebar.header("- Cell Controls & Sidebar Sync")
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
    avatar_icon = "CRITICAL"
elif current_soh <= 85.0 or slope_val <= -0.05:
    status_label = "Monitor Closely"
    badge_class = "badge-monitor"
    avatar_icon = "MONITOR"
else:
    status_label = "Healthy"
    badge_class = "badge-healthy"
    avatar_icon = "HEALTHY"

# Continuous SoH Color Gradient calculation (Emerald -> Amber -> Red)
soh_norm = np.clip((current_soh - 80.0) / 20.0, 0.0, 1.0)
if soh_norm >= 0.5:
    factor = (soh_norm - 0.5) * 2.0
    r_val = int(245 * (1 - factor) + 34 * factor)
    g_val = int(158 * (1 - factor) + 197 * factor)
    b_val = int(11 * (1 - factor) + 94 * factor)
else:
    factor = soh_norm * 2.0
    r_val = int(239 * (1 - factor) + 245 * factor)
    g_val = int(68 * (1 - factor) + 158 * factor)
    b_val = int(68 * (1 - factor) + 11 * factor)

soh_gradient_color = f"rgb({r_val}, {g_val}, {b_val})"

# Step 7 Pop-Up / Modal Detail View via st.dialog (Hardened & Error-Safe)
@st.dialog("Comprehensive Cell Telemetry & Model Details")
def show_cell_details_dialog(c_id, df_c, r_row):
    st.markdown(f"### Battery Cell `{c_id}` Deep-Dive Diagnostics")
    
    # Helper for safe column retrieval
    def get_field(field_name, default=0.0):
        try:
            if isinstance(r_row, pd.DataFrame) and len(r_row) > 0:
                return r_row[field_name].iloc[0]
            elif isinstance(r_row, pd.Series) and field_name in r_row:
                return r_row[field_name]
        except Exception:
            pass
        return default

    init_soh = float(df_c['soh_ground_truth'].iloc[0]) if len(df_c) > 0 else 100.0
    last_soh = float(df_c['soh_predicted'].iloc[-1]) if len(df_c) > 0 else 90.0
    slope_val_m = float(get_field('trend_slope', 0.0))
    likely_rul = int(get_field('rul_likely_cycles', 0))
    worst_rul = int(get_field('rul_worst_cycles', 0))
    best_rul = int(get_field('rul_best_cycles', 5000))
    top_d = str(get_field('top_driver', "cumulative_time_above_40C"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Cycles Monitored", f"{len(df_c)}")
        st.metric("Initial SoH", f"{init_soh:.2f}%")
    with c2:
        st.metric("Latest SoH", f"{last_soh:.2f}%")
        st.metric("Decline Velocity", f"{slope_val_m:.5f}%/cycle")
    with c3:
        st.metric("Likely RUL", f"{likely_rul} cycles")
        st.metric("P10-P90 RUL Spread", f"{worst_rul} – {best_rul}")
    
    st.markdown("---")
    st.markdown("#### Feature Stressors & Physics Model Assumptions")
    st.write(f"**Primary Degradation Stressor**: `{top_d}`")
    st.write("**Physics Degradation Engine**: Exponential decay ($k \\in [0.9, 1.1] \\times k_{base}$), thermal stress gate ($T > 40^\\circ\\text{C}$), high C-rate acceleration ($C > 1.5$).")
    
    st.markdown("---")
    st.markdown("#### Telemetry Snapshot (First 5 & Last 5 Cycles)")
    
    try:
        snap_df = pd.concat([df_c.head(5), df_c.tail(5)])
        target_cols = ['cycle_id', 'soh_predicted', 'temperature_max', 'cumulative_time_above_40C', 'c_rate', 'discharge_depth']
        valid_cols = [c for c in target_cols if c in snap_df.columns]
        
        if valid_cols:
            st.dataframe(snap_df[valid_cols], use_container_width=True)
        else:
            st.info("Telemetry snapshot data is empty or unavailable.")
    except Exception:
        st.warning("Unable to load telemetry snapshot table.")

st.sidebar.markdown("---")
st.sidebar.subheader("- Active Cell Profile")
st.sidebar.write(f"**Cell Identifier**: {avatar_icon} `{selected_cell}`")
st.sidebar.write(f"**Telemetry History**: `{len(df_cell)} cycles`")
st.sidebar.write(f"**Decline Velocity**: `{slope_val:.5f}% / cycle`")
st.sidebar.write(f"**Status Tier**: <span class=\"badge-pill {badge_class}\">{status_label}</span>", unsafe_allow_html=True)
if st.sidebar.button("View Cell Telemetry Details", use_container_width=True):
    show_cell_details_dialog(selected_cell, df_cell, cell_rul_row)

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
# 7. CHART 1: SOH DECAY TRAJECTORY (Step 4 Dark Floating Card Hover Popups)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-header">- State-of-Health (SoH) Decay Trajectory</div>', unsafe_allow_html=True)

df_cell['soh_delta'] = df_cell['soh_predicted'].diff().fillna(0.0)

fig_soh = go.Figure()

fig_soh.add_trace(go.Scatter(
    x=pd.concat([df_cell['cycle_id'], df_cell['cycle_id'][::-1]]),
    y=pd.concat([df_cell['soh_upper'], df_cell['soh_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(59, 130, 246, 0.12)',
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
    hovertemplate='<b>Cycle %{x}</b><br>Ground Truth: <b>%{y:.2f}%</b><extra></extra>',
    line=dict(color='#94a3b8', width=2, dash='dash')
))

fig_soh.add_trace(go.Scatter(
    x=df_cell['cycle_id'],
    y=df_cell['soh_predicted'],
    mode='lines',
    name='Predicted SoH (Model)',
    customdata=np.stack((df_cell['soh_delta'],), axis=-1),
    hovertemplate='<b>Cycle %{x}</b><br>Predicted SoH: <b>%{y:.2f}%</b><br><span style="color:#3b82f6; font-weight:700;">Delta: %{customdata[0]:+.3f}% / cycle</span><extra></extra>',
    line=dict(color='#3b82f6', width=3)
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
    plot_bgcolor='#12151c',
    hoverlabel=dict(
        bgcolor="#12151c",
        bordercolor="rgba(255,255,255,0.12)",
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#f8fafc"
    ),
    xaxis=dict(
        title="Cycle Count", 
        gridcolor='rgba(255,255,255,0.04)',
        zerolinecolor='rgba(255,255,255,0.08)'
    ),
    yaxis=dict(
        title="State-of-Health (%)", 
        gridcolor='rgba(255,255,255,0.04)',
        zerolinecolor='rgba(255,255,255,0.08)'
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
    st.markdown('<div class="section-header">- Cell Stressor Importance Breakdown</div>', unsafe_allow_html=True)
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
        hovertemplate='<b>%{y}</b><br>Importance: <b>%{x:.2%}</b><extra></extra>',
        marker=dict(
            color=driver_df_cell['Importance'],
            colorscale=[[0, '#1e293b'], [0.5, '#2563eb'], [1, '#3b82f6']],
            line=dict(color='rgba(255,255,255,0.08)', width=1)
        )
    ))
    fig_driver.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#12151c',
        hoverlabel=dict(
            bgcolor="#12151c",
            bordercolor="rgba(255,255,255,0.12)",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#f8fafc"
        ),
        xaxis=dict(
            title="Relative Stress Contribution", 
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.04)'
        ),
        yaxis=dict(autorange="reversed", gridcolor='rgba(255,255,255,0.04)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_driver, width='stretch')

with col_chart2:
    st.markdown('<div class="section-header">⏳ RUL Uncertainty Boundaries</div>', unsafe_allow_html=True)
    rul_chart_df = pd.DataFrame({
        'Scenario': ['Conservative (P10)', 'Likely (P50)', 'Optimistic (P90)'],
        'Cycles': [rul_worst, rul_likely, rul_best],
        'Color': ['#ef4444', '#f59e0b', '#22c55e']
    })
    
    fig_rul = go.Figure(go.Bar(
        x=rul_chart_df['Scenario'],
        y=rul_chart_df['Cycles'],
        text=rul_chart_df['Cycles'].astype(str) + " cycles",
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Remaining: <b>%{y} cycles</b><extra></extra>',
        marker=dict(
            color=rul_chart_df['Color'],
            line=dict(color='rgba(255,255,255,0.12)', width=1)
        )
    ))
    fig_rul.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#12151c',
        hoverlabel=dict(
            bgcolor="#12151c",
            bordercolor="rgba(255,255,255,0.12)",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#f8fafc"
        ),
        yaxis=dict(title="Cycles to 80% Threshold", gridcolor='rgba(255,255,255,0.04)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_rul, width='stretch')

# -----------------------------------------------------------------------------
# 9. AI SAFETY NARRATIVE DIAGNOSTIC SUMMARY (Step 5 Animated AI Thinking State)
# -----------------------------------------------------------------------------
st.markdown(f'''
<div class="section-header">
    - AI Safety Diagnostic Summary
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

narrative_placeholder = st.empty()
narrative_placeholder.markdown('''
<div class="ai-thinking-container">
    <div class="ai-orb-wrap">
        <div class="ai-orb-ring"></div>
        <div class="ai-thinking-orb"></div>
        <div class="ai-particle"></div>
        <div class="ai-particle"></div>
        <div class="ai-particle"></div>
        <div class="ai-particle"></div>
    </div>
    <div class="ai-thinking-text">
        <strong>AI Assistant is analyzing this cell...</strong><br>
        Synthesizing telemetry history, stress drivers, and risk guardrails
    </div>
</div>
''', unsafe_allow_html=True)

narrative_output = fetch_real_narrative(
    current_soh, slope_val, top_feat_name, rul_best, rul_likely, rul_worst
)
narrative_placeholder.empty()

st.markdown(f'<div class="narrative-box">{narrative_output}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. EXPANDABLE TECHNICAL DETAILS & PAGINATED TELEMETRY TABLE
# -----------------------------------------------------------------------------
with st.expander("Telemetry & Physics Degradation Model Parameters"):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
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

with st.expander("Paginated Telemetry Data Table"):
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
        if st.button("Next Page", disabled=(current_page == total_pages - 1), use_container_width=True):
            st.session_state["table_page"] += 1
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # Close main-animated-container
