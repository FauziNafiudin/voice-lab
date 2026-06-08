import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Voice Lab",
    page_icon="🎤",
    layout="wide"
)

# ========== CUSTOM CSS PREMIUM AUDIO STUDIO — WHITE THEME ==========
st.markdown("""
<style>
    /* ====== IMPORT COMFORTAA ====== */
    @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;500;600;700&display=swap');

    /* ====== BASE ====== */
    .stApp {
        background: #FFFFFF;
        color: #1A1A2E;
        font-family: 'Comfortaa', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp::after {
        content: '';
        position: fixed;
        top: -200px; left: -200px;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(0,150,80,0.03) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: #F5F7FA !important;
        border-right: 1px solid rgba(0,180,100,0.15);
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00C060, #00B4D8, #7C3AED);
    }

    /* ====== MAIN CONTENT ====== */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ====== TYPOGRAPHY ====== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Comfortaa', sans-serif;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    h2 {
        font-size: 1.55rem !important;
        color: #1A1A2E !important;
        margin-top: 2.2rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,0,0,0.08);
    }
    h2::before {
        content: '// ';
        color: #00C060;
        font-family: 'Comfortaa', monospace;
        font-size: 0.85rem;
        font-weight: 400;
    }
    h3 {
        font-size: 1.2rem !important;
        color: #3A4A60 !important;
        font-weight: 500;
    }

    /* ====== HERO BANNER ====== */
    .hero-banner {
        background: linear-gradient(135deg, #F0FFF8 0%, #E8F5FF 50%, #F5F0FF 100%);
        border: 1px solid rgba(0,200,100,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before, .hero-banner::after {
        content: '';
        position: absolute;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(0,200,100,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner::before {
        top: -80px; right: -80px;
    }
    .hero-banner::after {
        bottom: -50px; left: 30%;
    }
    .hero-tagline {
        font-family: 'Comfortaa', monospace;
        font-size: 0.75rem;
        color: #00A050;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        display: block;
    }
    .hero-title {
        font-family: 'Comfortaa', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #1A1A2E;
        line-height: 1.15;
        margin-bottom: 0.8rem;
    }
    .hero-title .green { color: #00A050; }
    .hero-desc {
        color: #556070;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 620px;
    }
    .waveform-deco {
        display: flex;
        align-items: center;
        gap: 3px;
        margin-top: 1.5rem;
    }
    .waveform-deco span {
        display: inline-block;
        width: 4px;
        background: linear-gradient(180deg, #00C060, #00B4D8);
        border-radius: 2px;
        opacity: 0.5;
        animation: wavepulse 1.2s ease-in-out infinite alternate;
    }
    @keyframes wavepulse {
        from { transform: scaleY(0.3); }
        to   { transform: scaleY(1); }
    }

    /* ====== STEP CARDS ====== */
    .step-card {
        background: #F8FAFC;
        border: 1px solid rgba(0,0,0,0.07);
        border-left: 3px solid #00C060;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .step-num {
        font-family: 'Comfortaa', monospace;
        font-size: 0.65rem;
        color: #00A050;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .step-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1A1A2E;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #556070;
        line-height: 1.5;
    }

    /* ====== SECTION PILL ====== */
    .section-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(0,180,80,0.08);
        border: 1px solid rgba(0,180,80,0.2);
        border-radius: 40px;
        padding: 0.2rem 0.8rem;
        font-family: 'Comfortaa', monospace;
        font-size: 0.7rem;
        color: #00A050;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* ====== BADGES ====== */
    .badge {
        display: inline-block;
        background: rgba(0,180,80,0.08);
        border: 1px solid rgba(0,180,80,0.22);
        color: #00A050;
        font-family: 'Comfortaa', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .badge.blue {
        background: rgba(0,150,200,0.08);
        border-color: rgba(0,150,200,0.22);
        color: #0090C0;
    }
    .badge.purple {
        background: rgba(100,50,200,0.08);
        border-color: rgba(100,50,200,0.22);
        color: #6030B0;
    }

    /* ====== BUTTONS ====== */
    .stButton > button {
        background: linear-gradient(135deg, #00C060, #009A4E) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0,180,80,0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00E676, #00C060) !important;
        box-shadow: 0 4px 16px rgba(0,200,80,0.25) !important;
        transform: translateY(-1px);
    }

    /* ====== METRICS ====== */
    [data-testid="stMetric"] {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,0,0,0.07) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Comfortaa', monospace !important;
        font-size: 0.6rem !important;
        color: #7A8FAA !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        color: #00A050 !important;
    }

    /* ====== EXPANDERS ====== */
    [data-testid="stExpander"] {
        background: #F8FAFC !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Comfortaa', sans-serif !important;
        font-weight: 500 !important;
        color: #556070 !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #00A050 !important;
    }

    /* ====== PROGRESS BAR ====== */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #00C060, #00B4D8) !important;
        border-radius: 999px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: rgba(0,0,0,0.07) !important;
        border-radius: 999px !important;
    }

    /* ====== CAPTIONS ====== */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'Comfortaa', monospace !important;
        font-size: 0.68rem !important;
        color: #8A9AB0 !important;
    }

    /* ====== ALERTS ====== */
    .stSuccess > div {
        background: rgba(0,180,80,0.07) !important;
        border-left: 3px solid #00C060 !important;
        border-radius: 8px !important;
        color: #006030 !important;
    }
    .stInfo > div {
        background: rgba(0,150,200,0.07) !important;
        border-left: 3px solid #00A0C8 !important;
        border-radius: 8px !important;
    }
    .stWarning > div {
        background: rgba(200,130,0,0.07) !important;
        border-left: 3px solid #C08000 !important;
        border-radius: 8px !important;
    }
    .stError > div {
        background: rgba(200,50,50,0.07) !important;
        border-left: 3px solid #C03030 !important;
        border-radius: 8px !important;
    }

    /* ====== AUDIO INPUT ====== */
    .stAudioInput > div {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,180,80,0.15) !important;
        border-radius: 12px !important;
    }

    /* ====== DIVIDER ====== */
    hr {
        border: none !important;
        border-top: 1px solid rgba(0,0,0,0.07) !important;
        margin: 2rem 0 !important;
    }

    /* ====== PAGE LINK ====== */
    [data-testid="stPageLink"] {
        background: #F0F4F8 !important;
        border: 1px solid rgba(0,180,80,0.15) !important;
        border-radius: 12px !important;
        transition: 0.2s !important;
    }
    [data-testid="stPageLink"]:hover {
        border-color: rgba(0,180,80,0.35) !important;
        box-shadow: 0 0 16px rgba(0,180,80,0.08) !important;
    }

    /* ====== SCROLLBAR ====== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F5F7FA; }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,180,80,0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0,180,80,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ========== MATPLOTLIB GLOBAL STYLE — WHITE THEME ==========
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CCCCCC',
    'axes.
