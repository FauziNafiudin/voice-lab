import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from pathlib import Path
from scipy.spatial.distance import cdist

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Voice Lab",
    page_icon="🎤",
    layout="wide"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;500;600;700&display=swap');

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
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Comfortaa', sans-serif; font-weight: 600; letter-spacing: -0.2px; }
    h2 { font-size: 1.55rem !important; color: #1A1A2E !important; margin-top: 2.2rem !important; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(0,0,0,0.08); }
    h2::before { content: '// '; color: #00C060; font-family: 'Comfortaa', monospace; font-size: 0.85rem; font-weight: 400; }
    h3 { font-size: 1.2rem !important; color: #3A4A60 !important; font-weight: 500; }

    .hero-banner {
        background: linear-gradient(135deg, #F0FFF8 0%, #E8F5FF 50%, #F5F0FF 100%);
        border: 1px solid rgba(0,200,100,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before, .hero-banner::after { content: ''; position: absolute; width: 250px; height: 250px; background: radial-gradient(circle, rgba(0,200,100,0.06) 0%, transparent 70%); pointer-events: none; }
    .hero-banner::before { top: -80px; right: -80px; }
    .hero-banner::after  { bottom: -50px; left: 30%; }
    .hero-tagline { font-family: 'Comfortaa', monospace; font-size: 0.75rem; color: #00A050; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.7rem; display: block; }
    .hero-title { font-family: 'Comfortaa', sans-serif; font-size: 2.6rem; font-weight: 700; color: #1A1A2E; line-height: 1.15; margin-bottom: 0.8rem; }
    .hero-title .green { color: #00A050; }
    .hero-desc { color: #556070; font-size: 1rem; line-height: 1.65; max-width: 620px; }
    .waveform-deco { display: flex; align-items: center; gap: 3px; margin-top: 1.5rem; }
    .waveform-deco span { display: inline-block; width: 4px; background: linear-gradient(180deg, #00C060, #00B4D8); border-radius: 2px; opacity: 0.5; animation: wavepulse 1.2s ease-in-out infinite alternate; }
    @keyframes wavepulse { from { transform: scaleY(0.3); } to { transform: scaleY(1); } }

    .step-card { background: #F8FAFC; border: 1px solid rgba(0,0,0,0.07); border-left: 3px solid #00C060; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; height: 100%; }
    .step-num   { font-family:'Comfortaa',monospace; font-size:0.65rem; color:#00A050; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.3rem; }
    .step-title { font-size:1rem; font-weight:600; color:#1A1A2E; margin-bottom:0.3rem; }
    .step-desc  { font-size:0.85rem; color:#556070; line-height:1.5; }

    .section-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(0,180,80,0.08); border: 1px solid rgba(0,180,80,0.2); border-radius: 40px; padding: 0.2rem 0.8rem; font-family: 'Comfortaa', monospace; font-size: 0.7rem; color: #00A050; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.6rem; }

    .badge        { display:inline-block; background:rgba(0,180,80,0.08); border:1px solid rgba(0,180,80,0.22); color:#00A050; font-family:'Comfortaa',monospace; font-size:0.62rem; letter-spacing:0.8px; text-transform:uppercase; padding:0.15rem 0.5rem; border-radius:4px; margin-right:0.4rem; margin-bottom:0.5rem; }
    .badge.blue   { background:rgba(0,150,200,0.08); border-color:rgba(0,150,200,0.22); color:#0090C0; }
    .badge.purple { background:rgba(100,50,200,0.08); border-color:rgba(100,50,200,0.22); color:#6030B0; }
    .badge.orange { background:rgba(255,107,53,0.08); border-color:rgba(255,107,53,0.22); color:#CC4400; }

    .stButton > button { background: linear-gradient(135deg, #00C060, #009A4E) !important; color: #FFFFFF !important; border: none !important; border-radius: 10px !important; font-family: 'Comfortaa', sans-serif !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 0.5rem 1.2rem !important; transition: 0.2s ease !important; box-shadow: 0 2px 8px rgba(0,180,80,0.2) !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #00E676, #00C060) !important; box-shadow: 0 4px 16px rgba(0,200,80,0.25) !important; transform: translateY(-1px); }

    [data-testid="stMetric"] { background: #F0F4F8 !important; border: 1px solid rgba(0,0,0,0.07) !important; border-radius: 12px !important; padding: 0.8rem 1rem !important; }
    [data-testid="stMetricLabel"] { font-family: 'Comfortaa', monospace !important; font-size: 0.6rem !important; color: #7A8FAA !important; letter-spacing: 1.2px !important; text-transform: uppercase !important; }
    [data-testid="stMetricValue"] { font-family: 'Comfortaa', sans-serif !important; font-weight: 700 !important; font-size: 1.8rem !important; color: #00A050 !important; }

    [data-testid="stExpander"] { background: #F8FAFC !important; border: 1px solid rgba(0,0,0,0.08) !important; border-radius: 12px !important; overflow: hidden; }
    [data-testid="stExpander"] summary { font-family: 'Comfortaa', sans-serif !important; font-weight: 500 !important; color: #556070 !important; }
    [data-testid="stExpander"] summary:hover { color: #00A050 !important; }

    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #00C060, #00B4D8) !important; border-radius: 999px !important; }
    [data-testid="stProgressBar"] > div { background: rgba(0,0,0,0.07) !important; border-radius: 999px !important; }

    .stCaption, [data-testid="stCaptionContainer"] { font-family: 'Comfortaa', monospace !important; font-size: 0.68rem !important; color: #8A9AB0 !important; }

    .stSuccess > div { background:rgba(0,180,80,0.07) !important; border-left:3px solid #00C060 !important; border-radius:8px !important; color:#006030 !important; }
    .stInfo    > div { background:rgba(0,150,200,0.07) !important; border-left:3px solid #00A0C8 !important; border-radius:8px !important; }
    .stWarning > div { background:rgba(200,130,0,0.07)  !important; border-left:3px solid #C08000 !important; border-radius:8px !important; }
    .stError   > div { background:rgba(200,50,50,0.07)  !important; border-left:3px solid #C03030 !important; border-radius:8px !important; }
    .stAudioInput > div { background: #F0F4F8 !important; border: 1px solid rgba(0,180,80,0.15) !important; border-radius: 12px !important; }
    hr { border:none !important; border-top:1px solid rgba(0,0,0,0.07) !important; margin:2rem 0 !important; }

    [data-testid="stPageLink"] { background: #F0F4F8 !important; border: 1px solid rgba(0,180,80,0.15) !important; border-radius: 12px !important; transition: 0.2s !important; }
    [data-testid="stPageLink"]:hover { border-color: rgba(0,180,80,0.35) !important; box-shadow: 0 0 16px rgba(0,180,80,0.08) !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F5F7FA; }
    ::-webkit-scrollbar-thumb { background: rgba(0,180,80,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,180,80,0.4); }

    /* ====== SIMILARITY CARD ====== */
    .sim-card { background: linear-gradient(135deg, #F0FFF8, #E8F5FF); border: 1px solid rgba(0,180,80,0.18); border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
    .sim-label { font-family: 'Comfortaa', monospace; font-size: 0.62rem; color: #7A9AB0; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.3rem; }
    .sim-score { font-size: 2.2rem; font-weight: 700; color: #00A050; font-family: 'Comfortaa', sans-serif; }
    .sim-bar-bg { height: 8px; background: rgba(0,0,0,0.07); border-radius: 999px; margin-top: 0.6rem; overflow: hidden; }
    .sim-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00C060, #00B4D8); transition: width 0.5s ease; }

    /* ====== METRIK DETAIL CARD ====== */
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin: 0.8rem 0; }
    .metric-item { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 0.7rem 0.9rem; }
    .metric-item-label { font-family: 'Comfortaa', monospace; font-size: 0.58rem; color: #8A9AB0; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.2rem; }
    .metric-item-value { font-family: 'Comfortaa', sans-serif; font-size: 1.15rem; font-weight: 700; color: #1A1A2E; }
    .metric-item-sub { font-family: 'Comfortaa', monospace; font-size: 0.6rem; color: #8A9AB0; margin-top: 0.1rem; }

    /* ====== FORMULA BOX ====== */
    .formula-box { background: #F0F4F8; border: 1px solid rgba(0,0,0,0.08); border-left: 3px solid #6030B0; border-radius: 10px; padding: 0.8rem 1rem; font-family: 'Comfortaa', monospace; font-size: 0.78rem; color: #3A4A60; line-height: 1.8; margin: 0.6rem 0; }
    .formula-box .fkey { color: #00A050; font-weight: 700; }
    .formula-box .fval { color: #6030B0; }
</style>
""", unsafe_allow_html=True)

# ========== MATPLOTLIB GLOBAL STYLE ==========
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#444444',
    'axes.titlecolor': '#222222',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'grid.color': '#DDDDDD',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'text.color': '#444444',
    'font.family': 'monospace',
    'lines.linewidth': 1.4,
})

# ============================================================
# ========== FUNGSI MFCC + DTW ==========================
# ============================================================

def compute_mfcc_features(file_path, sr=16000, n_mfcc=13):
    """Ekstrak MFCC + Delta MFCC. Return (features, y, sr_actual) atau (None, None, None)."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        features = np.vstack([mfcc, delta]).T   # (T, n_mfcc*2)
        return features, y, sr
    except Exception as e:
        st.error(f"Gagal memproses {file_path}: {e}")
        return None, None, None


def dtw_full(seq1, seq2):
    """
    Hitung DTW lengkap.
    Return dict berisi:
      - cost_matrix      : (n, m) cosine distance antar semua pasang frame
      - dtw_matrix       : (n+1, m+1) accumulated cost
      - path             : list of (i,j) optimal warping path
      - raw_distance     : DTW distance mentah (akumulasi di endpoint)
      - normalized_dist  : raw / (n + m)   → metrik utama
      - n_frames_ref     : jumlah frame sekuens referensi
      - n_frames_test    : jumlah frame sekuens test
      - mean_frame_cost  : rata-rata cosine distance di sepanjang path
    """
    cost_matrix = cdist(seq1, seq2, metric='cosine')
    n, m = cost_matrix.shape

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    # Traceback untuk warping path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        moves = {
            (i - 1, j - 1): dtw_matrix[i - 1, j - 1],
            (i - 1, j):     dtw_matrix[i - 1, j],
            (i, j - 1):     dtw_matrix[i, j - 1],
        }
        best = min(moves, key=moves.get)
        i, j = best
    path.reverse()

    raw_distance = float(dtw_matrix[n, m])
    normalized   = raw_distance / (n + m)
    path_costs   = [cost_matrix[p[0], p[1]] for p in path]
    mean_cost    = float(np.mean(path_costs)) if path_costs else 0.0

    return {
        "cost_matrix":     cost_matrix,
        "dtw_matrix":      dtw_matrix,
        "path":            path,
        "raw_distance":    raw_distance,
        "normalized_dist": float(normalized),
        "n_frames_ref":    n,
        "n_frames_test":   m,
        "mean_frame_cost": mean_cost,
    }


def dtw_to_similarity(dtw_dist, scale=3.5):
    """exp(-scale × normalized_dist) → [0, 1]"""
    return float(np.exp(-scale * dtw_dist))


def get_mfcc_features(file_path, min_frames=10):
    """Wrapper aman; return (features, y, sr) atau (None, None, None)."""
    features, y, sr = compute_mfcc_features(file_path)
    if features is None:
        return None, None, None
    if len(features) < min_frames:
        st.warning("Rekaman terlalu pendek. Minimal ±0.3 detik.")
        return None, None, None
    return features, y, sr


def compute_similarity_full(feat_ref, feat_test):
    """Jalankan DTW penuh; return (similarity, dtw_result_dict)."""
    if feat_ref is None or feat_test is None:
        return 0.0, None
    result = dtw_full(feat_ref, feat_test)
    sim    = dtw_to_similarity(result["normalized_dist"], scale=3.5)
    return sim, result


def similarity_label(score):
    if score >= 0.65:
        return "✅", "Sangat mirip — tiruan kucing yang sangat bagus!", "#006030"
    elif score >= 0.45:
        return "🔊", "Cukup mirip — ada nuansa suara kucing.", "#005080"
    elif score >= 0.25:
        return "🐾", "Sedikit mirip — masih terasa suara manusia.", "#7A5800"
    else:
        return "❌", "Sangat berbeda — ini suara manusia normal.", "#800000"


# ============================================================
# ========== RENDER DETAIL METRIK ============================
# ============================================================

def render_metric_detail(label, dtw_result, feat_ref, feat_test, y_ref, y_test, sr_val, sim_score):
    """
    Tampilkan breakdown lengkap metrik DTW di dalam expander.
    label    : 'Manusia 1' / 'Manusia 2' / 'Cat2.mp3'
    """
    if dtw_result is None:
        return

    nd   = dtw_result["normalized_dist"]
    raw  = dtw_result["raw_distance"]
    nref = dtw_result["n_frames_ref"]
    ntest= dtw_result["n_frames_test"]
    mcost= dtw_result["mean_frame_cost"]
    path = dtw_result["path"]
    cost_matrix = dtw_result["cost_matrix"]

    with st.expander(f"🔬 Detail Perhitungan Metrik — {label}", expanded=False):

        # ── 1. FORMULA & ANGKA ─────────────────────────────────────
        st.markdown("**📐 Rumus & Hasil Numerik**")
        st.markdown(f"""
        <div class="formula-box">
            <span class="fkey">Step 1 — Cost Matrix</span><br>
            &nbsp;&nbsp;cost[i,j] = cosine_distance(MFCC_ref[i], MFCC_test[j])<br>
            &nbsp;&nbsp;= 1 − (MFCC_ref[i] · MFCC_test[j]) / (‖MFCC_ref[i]‖ × ‖MFCC_test[j]‖)<br>
            &nbsp;&nbsp;Shape: <span class="fval">{nref} × {ntest} = {nref*ntest:,} pasang frame</span><br><br>

            <span class="fkey">Step 2 — DTW Accumulated Cost</span><br>
            &nbsp;&nbsp;DTW[i,j] = cost[i,j] + min(DTW[i−1,j], DTW[i,j−1], DTW[i−1,j−1])<br>
            &nbsp;&nbsp;Raw DTW distance (endpoint): <span class="fval">{raw:.4f}</span><br><br>

            <span class="fkey">Step 3 — Normalisasi</span><br>
            &nbsp;&nbsp;normalized = raw / (n_ref + n_test)<br>
            &nbsp;&nbsp;= {raw:.4f} / ({nref} + {ntest}) = <span class="fval">{nd:.4f}</span><br><br>

            <span class="fkey">Step 4 — Konversi ke Similarity</span><br>
            &nbsp;&nbsp;similarity = exp(−3.5 × normalized)<br>
            &nbsp;&nbsp;= exp(−3.5 × {nd:.4f}) = <span class="fval">{sim_score:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. GRID METRIK ─────────────────────────────────────────
        st.markdown("**📊 Ringkasan Metrik**")
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-item-label">Skor Kemiripan</div>
                <div class="metric-item-value" style="color:#00A050">{sim_score:.4f}</div>
                <div class="metric-item-sub">exp(−3.5 × DTW_norm)</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">DTW Normalized</div>
                <div class="metric-item-value" style="color:#6030B0">{nd:.4f}</div>
                <div class="metric-item-sub">raw / (n_ref + n_test)</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">DTW Raw Distance</div>
                <div class="metric-item-value" style="color:#0090C0">{raw:.4f}</div>
                <div class="metric-item-sub">Akumulasi cost di endpoint</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Frame Referensi</div>
                <div class="metric-item-value">{nref}</div>
                <div class="metric-item-sub">Frame MFCC Cat.mp3</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Frame Test</div>
                <div class="metric-item-value">{ntest}</div>
                <div class="metric-item-sub">Frame MFCC rekaman</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Mean Frame Cost</div>
                <div class="metric-item-value" style="color:#CC4400">{mcost:.4f}</div>
                <div class="metric-item-sub">Rata-rata cosine dist di path</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-item-label">Panjang Warping Path</div>
                <div class="metric-item-value">{len(path)}</div>
                <div class="metric-item-sub">Jumlah langkah alignment</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Rasio Durasi</div>
                <div class="metric-item-value">{ntest/nref:.2f}×</div>
                <div class="metric-item-sub">n_test / n_ref</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Dimensi Fitur</div>
                <div class="metric-item-value">26</div>
                <div class="metric-item-sub">13 MFCC + 13 Delta</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3. HEATMAP DTW COST MATRIX ─────────────────────────────
        st.markdown("**🗺️ DTW Cost Matrix Heatmap**")
        st.caption("Warna gelap = cosine distance rendah (lebih mirip). Garis putih = warping path optimal.")

        # Subsample cost matrix agar cepat dirender (max 150×150)
        max_dim = 150
        step_r  = max(1, nref  // max_dim)
        step_c  = max(1, ntest // max_dim)
        cm_sub  = cost_matrix[::step_r, ::step_c]

        fig_heat, ax_heat = plt.subplots(figsize=(7, 4))
        im = ax_heat.imshow(
            cm_sub.T,
            origin='lower',
            aspect='auto',
            cmap='magma_r',
            vmin=0, vmax=min(1.0, cost_matrix.max()),
        )
        plt.colorbar(im, ax=ax_heat, label='Cosine Distance', fraction=0.03)

        # Plot warping path (subsample juga)
        path_arr = np.array(path)
        if len(path_arr) > 0:
            px = path_arr[:, 0] / step_r
            py = path_arr[:, 1] / step_c
            ax_heat.plot(px, py, color='white', linewidth=1.2, alpha=0.85, label='Warping Path')
            ax_heat.legend(fontsize=7, loc='upper left')

        ax_heat.set_xlabel(f"Frame Referensi (Cat.mp3, subsampled /{step_r})")
        ax_heat.set_ylabel(f"Frame Test ({label}, subsampled /{step_c})")
        ax_heat.set_title("DTW Cost Matrix (cosine distance per frame-pair)")
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        # ── 4. DISTRIBUSI COST SEPANJANG PATH ─────────────────────
        st.markdown("**📈 Distribusi Cosine Distance Sepanjang Warping Path**")
        st.caption("Semakin rendah dan merata → suara semakin mirip di setiap titik.")

        path_costs = np.array([cost_matrix[p[0], p[1]] for p in path])
        x_axis     = np.linspace(0, 100, len(path_costs))  # persentase path

        fig_path, ax_path = plt.subplots(figsize=(7, 2.8))
        ax_path.fill_between(x_axis, path_costs, alpha=0.15, color='#6030B0')
        ax_path.plot(x_axis, path_costs, color='#6030B0', linewidth=1.2)
        ax_path.axhline(mcost, color='#FF6B35', linewidth=1, linestyle='--', label=f'Mean = {mcost:.3f}')
        ax_path.set_xlabel("Posisi di Warping Path (%)")
        ax_path.set_ylabel("Cosine Distance")
        ax_path.set_title("Cosine Distance Tiap Langkah Warping Path")
        ax_path.legend(fontsize=8)
        ax_path.grid(True)
        fig_path.tight_layout()
        st.pyplot(fig_path)
        plt.close(fig_path)

        # ── 5. PERBANDINGAN MFCC ───────────────────────────────────
        st.markdown("**🎨 Perbandingan MFCC: Cat.mp3 vs Rekaman**")
        st.caption("13 koefisien MFCC statis. Pola yang mirip → rekaman akustik mirip kucing.")

        # Ambil hanya MFCC statis (13 pertama), bukan delta
        mfcc_ref_only  = feat_ref[:, :13].T   # (13, T_ref)
        mfcc_test_only = feat_test[:, :13].T  # (13, T_test)

        fig_mfcc, (axm1, axm2) = plt.subplots(2, 1, figsize=(9, 5), sharex=False)

        img1 = axm1.imshow(
            mfcc_ref_only,
            aspect='auto', origin='lower', cmap='viridis',
            extent=[0, mfcc_ref_only.shape[1], 0.5, 13.5]
        )
        axm1.set_title("MFCC — Cat.mp3 (Referensi)", fontsize=10)
        axm1.set_ylabel("Koefisien MFCC")
        axm1.set_xlabel("Frame")
        plt.colorbar(img1, ax=axm1, fraction=0.02)

        img2 = axm2.imshow(
            mfcc_test_only,
            aspect='auto', origin='lower', cmap='viridis',
            extent=[0, mfcc_test_only.shape[1], 0.5, 13.5]
        )
        axm2.set_title(f"MFCC — {label} (Rekaman)", fontsize=10)
        axm2.set_ylabel("Koefisien MFCC")
        axm2.set_xlabel("Frame")
        plt.colorbar(img2, ax=axm2, fraction=0.02)

        fig_mfcc.tight_layout()
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)

        # ── 6. PROFIL RATA-RATA MFCC ──────────────────────────────
        st.markdown("**📉 Profil Rata-rata MFCC (per Koefisien)**")
        st.caption("Bila kedua garis berdekatan → karakteristik spektral mirip.")

        mean_ref  = feat_ref[:, :13].mean(axis=0)
        mean_test = feat_test[:, :13].mean(axis=0)
        k_idx     = np.arange(1, 14)

        fig_prof, ax_prof = plt.subplots(figsize=(7, 3))
        ax_prof.plot(k_idx, mean_ref,  'o-', color='#00A050', linewidth=1.5, markersize=5, label='Cat.mp3 (Referensi)')
        ax_prof.plot(k_idx, mean_test, 's--', color='#FF6B35', linewidth=1.5, markersize=5, label=f'{label} (Rekaman)')
        ax_prof.fill_between(k_idx, mean_ref, mean_test, alpha=0.07, color='#888888')
        ax_prof.set_xlabel("Koefisien MFCC ke-")
        ax_prof.set_ylabel("Nilai Rata-rata")
        ax_prof.set_title("Profil Rata-rata MFCC: Referensi vs Rekaman")
        ax_prof.set_xticks(k_idx)
        ax_prof.legend(fontsize=8)
        ax_prof.grid(True)
        fig_prof.tight_layout()
        st.pyplot(fig_prof)
        plt.close(fig_prof)

        # ── 7. INTERPRETASI TABEL ─────────────────────────────────
        st.markdown("**📋 Panduan Interpretasi Skor**")
        st.markdown("""
| Skor | DTW Normalized | Interpretasi |
|------|---------------|--------------|
| ≥ 0.65 | ≤ 0.11 | Sangat mirip kucing secara akustik |
| 0.45 – 0.65 | 0.11 – 0.18 | Cukup mirip, ada nuansa kucing |
| 0.25 – 0.45 | 0.18 – 0.28 | Sedikit mirip, masih dominan manusia |
| < 0.25 | > 0.28 | Sangat berbeda, suara manusia normal |
        """)


# ============================================================
# ========== HERO HEADER =====================================
# ============================================================
st.markdown("""
<div class="hero-banner">
    <span class="hero-tagline">🎤 Interactive Audio Lab</span>
    <div class="hero-title">Voice <span class="green">Lab</span></div>
    <p class="hero-desc">
        Jelajahi dunia <strong style="color:#1A1A2E">audio processing</strong> secara interaktif —
        dari gelombang sintetik sampai pengenalan suara berbasis MFCC + DTW.
        Rekam, bandingkan, dan rasakan sendiri cara komputer "mendengar".
    </p>
    <div class="waveform-deco">
        <span style="height:8px;  animation-delay:0s"></span>
        <span style="height:22px; animation-delay:0.10s"></span>
        <span style="height:36px; animation-delay:0.20s"></span>
        <span style="height:14px; animation-delay:0.30s"></span>
        <span style="height:44px; animation-delay:0.40s"></span>
        <span style="height:26px; animation-delay:0.50s"></span>
        <span style="height:38px; animation-delay:0.60s"></span>
        <span style="height:12px; animation-delay:0.70s"></span>
        <span style="height:30px; animation-delay:0.80s"></span>
        <span style="height:46px; animation-delay:0.90s"></span>
        <span style="height:18px; animation-delay:1.00s"></span>
        <span style="height:32px; animation-delay:0.05s"></span>
        <span style="height:42px; animation-delay:0.15s"></span>
        <span style="height:10px; animation-delay:0.25s"></span>
        <span style="height:28px; animation-delay:0.35s"></span>
        <span style="height:40px; animation-delay:0.45s"></span>
        <span style="height:20px; animation-delay:0.55s"></span>
        <span style="height:34px; animation-delay:0.65s"></span>
        <span style="height:16px; animation-delay:0.75s"></span>
        <span style="height:48px; animation-delay:0.85s"></span>
        <span style="height:24px; animation-delay:0.95s"></span>
        <span style="height:38px; animation-delay:0.08s"></span>
        <span style="height:12px; animation-delay:0.18s"></span>
        <span style="height:44px; animation-delay:0.28s"></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ========== BAB 01: APA ITU AUDIO? ==========================
# ============================================================
st.markdown('<div class="section-pill">📡 Bab 01</div>', unsafe_allow_html=True)
st.header("Apa Itu Audio?")
st.markdown("""
<p style="color:#556070; font-size:0.95rem; line-height:1.8; margin-bottom:1.5rem">
Audio adalah <strong style="color:#1A1A2E">getaran udara</strong> yang ditangkap telinga dan direpresentasikan
sebagai sinyal digital oleh komputer. Setiap suara punya <strong style="color:#00A050">bentuk gelombang (waveform)</strong>
yang unik — misalnya, suara kucing mengeong punya pola yang sama sekali berbeda dengan suara manusia.
</p>
""", unsafe_allow_html=True)

st.subheader("Contoh Waveform Sinyal Sintetik")
st.markdown('<span class="badge">Synthetic</span><span class="badge blue">400 Hz + 800 Hz</span>', unsafe_allow_html=True)
st.caption("Gelombang sintetik sangat teratur dan mudah dipelajari — penjumlahan dua frekuensi berbeda.")

sr_synth = 16000
t_synth  = np.linspace(0, 0.1, int(sr_synth * 0.1))
y_synth  = np.sin(2 * np.pi * 400 * t_synth) + 0.5 * np.sin(2 * np.pi * 800 * t_synth)

fig_synth, ax_synth = plt.subplots(figsize=(9, 2.5))
ax_synth.plot(t_synth, y_synth, color='#00A050', linewidth=1.2, alpha=0.9)
ax_synth.fill_between(t_synth, y_synth, alpha=0.08, color='#00A050')
ax_synth.set_title("Waveform Sinyal Sintetik (400 Hz + 800 Hz) — 0.1 detik")
ax_synth.set_xlabel("Waktu (detik)")
ax_synth.set_ylabel("Amplitudo")
ax_synth.grid(True)
fig_synth.tight_layout()
st.pyplot(fig_synth)
plt.close(fig_synth)
st.caption("Gelombang ini sangat halus dan periodik, berbeda dengan suara alami yang lebih kompleks.")

with st.expander("🐱 Contoh Asli: Waveform Suara Kucing"):
    cat_file = "Cat.mp3"
    if os.path.exists(cat_file):
        st.audio(cat_file, format="audio/mpeg")
        st.caption("▶️ Dengarkan suara kucing asli.")
        y_cat, sr_cat = librosa.load(cat_file, sr=16000)
        duration = len(y_cat) / sr_cat
        sample_limit = min(2 * sr_cat, len(y_cat))
        t_cat = np.linspace(0, sample_limit / sr_cat, sample_limit)
        fig_cat, ax_cat = plt.subplots(figsize=(9, 2.5))
        ax_cat.plot(t_cat, y_cat[:sample_limit], color='#0090C0', linewidth=0.8, alpha=0.9)
        ax_cat.fill_between(t_cat, y_cat[:sample_limit], alpha=0.07, color='#0090C0')
        ax_cat.set_title("Waveform Asli Cat.mp3 (2 detik pertama)")
        ax_cat.set_xlabel("Waktu (detik)")
        ax_cat.set_ylabel("Amplitudo")
        ax_cat.grid(True)
        fig_cat.tight_layout()
        st.pyplot(fig_cat)
        plt.close(fig_cat)
        st.caption(f"Durasi total: {duration:.2f} detik. Bentuknya lebih acak dibanding sinyal sintetik.")
    else:
        st.warning(f"File {cat_file} tidak ditemukan.")
        st.info("Pastikan file Cat.mp3 berada di folder yang sama dengan aplikasi ini.")

# ============================================================
# ========== BAB 02: BAGAIMANA KOMPUTER PROSES SUARA =========
# ============================================================
st.markdown('<div class="section-pill" style="margin-top:2rem">🔬 Bab 02</div>', unsafe_allow_html=True)
st.header("Bagaimana Komputer Memproses Suara?")
st.markdown('<p style="color:#556070; font-size:0.93rem; margin-bottom:1.5rem">Komputer mengubah gelombang suara menjadi angka, lalu menganalisisnya melalui beberapa tahap berikut.</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="step-card"><div class="step-num">STEP 01</div><div class="step-title">Sampling</div><div class="step-desc">Mengubah gelombang analog menjadi angka diskrit pada interval tetap</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="step-card"><div class="step-num">STEP 02</div><div class="step-title">Waveform</div><div class="step-desc">Visualisasi deretan angka sampel sebagai gelombang digital</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="step-card"><div class="step-num">STEP 03</div><div class="step-title">Windowing</div><div class="step-desc">Memotong suara panjang menjadi frame pendek 20–50 ms</div></div>', unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown('<div class="step-card"><div class="step-num">STEP 04</div><div class="step-title">FFT</div><div class="step-desc">Mengubah frame dari domain waktu ke spektrum frekuensi</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="step-card"><div class="step-num">STEP 05</div><div class="step-title">Spektrogram</div><div class="step-desc">Melihat bagaimana frekuensi berubah seiring waktu (2D)</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown('<div class="step-card"><div class="step-num">STEP 06</div><div class="step-title">MFCC + DTW</div><div class="step-desc">Fitur akustik ringkas + Dynamic Time Warping untuk perbandingan suara</div></div>', unsafe_allow_html=True)

# ── LOAD FILE KUCING ──────────────────────────────────────
cat_file = "Cat.mp3"
if os.path.exists(cat_file):
    y_full, sr = librosa.load(cat_file, sr=16000)
    durasi_total = len(y_full) / sr
    cat_available = True
    n_pendek = int(0.05 * sr);  y_pendek = y_full[:n_pendek]; t_pendek = np.linspace(0, 0.05, n_pendek)
    n_panjang= int(2.0  * sr);  y_panjang= y_full[:n_panjang]; t_panjang= np.linspace(0, 2.0, n_panjang)
else:
    cat_available = False
    st.warning("File Cat.mp3 tidak ditemukan. Contoh asli tidak tersedia.")

COLORS = ['#00A050', '#FF6B35', '#6030B0', '#C08000', '#0090C0']

# ── LANGKAH 1: SAMPLING ───────────────────────────────────
st.subheader("1 — Sampling")
st.markdown('<span class="badge">Analog → Digital</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#1A1A2E">Sampling</strong> mengambil nilai amplitudo pada interval waktu tetap. <strong style="color:#00A050">Sample rate</strong> (Hz) = jumlah sampel per detik.</p>', unsafe_allow_html=True)

sr_sintetis = 200
t_cont = np.linspace(0, 0.1, 1000);  y_cont = np.sin(2 * np.pi * 50 * t_cont)
t_sample_sint = np.linspace(0, 0.1, int(sr_sintetis * 0.1))
y_sample_sint = np.sin(2 * np.pi * 50 * t_sample_sint)

fig1, ax1 = plt.subplots(figsize=(9, 3))
ax1.plot(t_cont, y_cont, color='#0090C0', alpha=0.4, linewidth=1.5, label='Gelombang analog (ilustrasi)')
ml, sl, bl = ax1.stem(t_sample_sint, y_sample_sint, linefmt='#00A050', markerfmt='o', basefmt=' ', label=f'Sampel ({sr_sintetis} Hz)')
sl.set_linewidth(1); ml.set_color('#00A050'); ml.set_markersize(5)
ax1.legend(fontsize=8); ax1.grid(True); fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)
st.caption("Titik hijau adalah hasil sampling. Komputer menyimpan angka-angka ini.")

if cat_available:
    with st.expander("🐱 Sampling pada suara kucing asli (50 ms)"):
        sr_ilustrasi = 200; step = sr // sr_ilustrasi
        t_sample_kucing = t_pendek[::step]; y_sample_kucing = y_pendek[::step]
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        ax2.plot(t_pendek, y_pendek, color='#0090C0', alpha=0.3, linewidth=1.2, label='Gelombang asli')
        ml2, sl2, bl2 = ax2.stem(t_sample_kucing, y_sample_kucing, linefmt='#00A050', markerfmt='o', basefmt=' ', label=f'Sampel ({sr_ilustrasi} Hz)')
        sl2.set_linewidth(1); ml2.set_color('#00A050'); ml2.set_markersize(5)
        ax2.set_xlabel("Waktu (detik)"); ax2.set_ylabel("Amplitudo"); ax2.set_title("Sampling suara kucing (50 ms)")
        ax2.legend(fontsize=8); ax2.grid(True); fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
        st.caption(f"{len(y_sample_kucing)} sampel disimpan komputer.")
        st.session_state['y_sample_kucing'] = y_sample_kucing
        st.session_state['t_sample_kucing'] = t_sample_kucing
        st.session_state['sr_ilustrasi']    = sr_ilustrasi

# ── LANGKAH 2: WAVEFORM ───────────────────────────────────
st.subheader("2 — Representasi Digital (Waveform)")
st.markdown('<span class="badge">Angka → Visualisasi</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8">Angka-angka hasil sampling diplot → <strong style="color:#00A050">Waveform</strong>.</p>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(9, 2.5))
ax3.plot(t_sample_sint, y_sample_sint, color='#6030B0', marker='o', markersize=4, linewidth=1.2)
ax3.fill_between(t_sample_sint, y_sample_sint, alpha=0.06, color='#6030B0')
ax3.set_title("Waveform sinyal sinus (hasil sampling 200 Hz)"); ax3.set_xlabel("Waktu (detik)"); ax3.set_ylabel("Amplitudo digital"); ax3.grid(True)
fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)
st.caption("Garis menghubungkan titik-titik sampling — inilah representasi digitalnya.")

if cat_available and 'y_sample_kucing' in st.session_state:
    with st.expander("🐱 Waveform dari sampling suara kucing"):
        y_samp = st.session_state['y_sample_kucing']; t_samp = st.session_state['t_sample_kucing']; sr_ilust = st.session_state['sr_ilustrasi']
        fig4, ax4 = plt.subplots(figsize=(9, 2.5))
        ax4.plot(t_samp, y_samp, color='#6030B0', marker='o', markersize=4, linewidth=1.2)
        ax4.fill_between(t_samp, y_samp, alpha=0.06, color='#6030B0')
        ax4.set_title(f"Waveform suara kucing ({len(y_samp)} sampel, {sr_ilust} Hz)"); ax4.set_xlabel("Waktu (detik)"); ax4.set_ylabel("Amplitudo digital"); ax4.grid(True)
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

# ── LANGKAH 3: WINDOWING ──────────────────────────────────
st.subheader("3 — Windowing")
st.markdown('<span class="badge">Segmentasi</span><span class="badge blue">Frame 25 ms</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8">Suara panjang dipotong jadi <strong style="color:#00A050">frame pendek 20–50 ms</strong> yang tumpang tindih.</p>', unsafe_allow_html=True)

durasi_sint = 0.5; t_sint = np.linspace(0, durasi_sint, int(16000 * durasi_sint)); y_sint = np.sin(2 * np.pi * 440 * t_sint)
frame_len = int(0.025 * 16000); hop_len = int(0.010 * 16000)

fig5, ax5 = plt.subplots(figsize=(9, 3))
ax5.plot(t_sint, y_sint, color='#0090C0', alpha=0.5, linewidth=1)
for i in range(5):
    start = i * hop_len
    if start + frame_len < len(t_sint):
        ax5.axvspan(t_sint[start], t_sint[start + frame_len], alpha=0.14, color=COLORS[i % len(COLORS)])
ax5.set_xlabel("Waktu (detik)"); ax5.set_title("Windowing — setiap warna adalah frame 25 ms"); ax5.grid(True)
fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)
st.caption("Setiap window akan dianalisis dengan FFT untuk melihat frekuensi pada saat itu.")

if cat_available:
    with st.expander("🐱 Windowing pada suara kucing (2 detik pertama)"):
        fig6, ax6 = plt.subplots(figsize=(9, 3))
        ax6.plot(t_panjang, y_panjang, color='#0090C0', alpha=0.45, linewidth=0.8)
        for i in range(8):
            start = i * hop_len
            if start + frame_len < len(t_panjang):
                ax6.axvspan(t_panjang[start], t_panjang[start + frame_len], alpha=0.12, color=COLORS[i % len(COLORS)])
        ax6.set_xlabel("Waktu (detik)"); ax6.set_title("Windowing suara kucing (25 ms, overlap 10 ms)"); ax6.grid(True)
        fig6.tight_layout(); st.pyplot(fig6); plt.close(fig6)

# ── LANGKAH 4: FFT ────────────────────────────────────────
st.subheader("4 — FFT (Fast Fourier Transform)")
st.markdown('<span class="badge">Waktu → Frekuensi</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">FFT</strong> mengubah frame ke spektrum frekuensi.</p>', unsafe_allow_html=True)

fs_demo = 16000; frame_demo = np.linspace(0, 0.025, int(0.025 * fs_demo))
y_demo = np.sin(2 * np.pi * 440 * frame_demo) + 0.5 * np.sin(2 * np.pi * 880 * frame_demo)
fft_vals = np.fft.fft(y_demo); freqs = np.fft.fftfreq(len(y_demo), 1 / fs_demo)
magnitude = np.abs(fft_vals[:len(fft_vals) // 2]); freqs_pos = freqs[:len(freqs) // 2]

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(9, 3))
ax7a.plot(frame_demo, y_demo, color='#0090C0', linewidth=1.2)
ax7a.fill_between(frame_demo, y_demo, alpha=0.07, color='#0090C0')
ax7a.set_title("Frame 25 ms (domain waktu)"); ax7a.set_xlabel("Waktu (detik)"); ax7a.grid(True)
ax7b.plot(freqs_pos, magnitude, color='#FF6B35', linewidth=1.2)
ax7b.fill_between(freqs_pos, magnitude, alpha=0.09, color='#FF6B35')
ax7b.set_title("Spektrum frekuensi (FFT)"); ax7b.set_xlabel("Frekuensi (Hz)"); ax7b.set_xlim(0, 2000); ax7b.grid(True)
plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)
st.caption("Puncak di 440 Hz dan 880 Hz terlihat jelas di spektrum.")

if cat_available:
    with st.expander("🐱 FFT pada satu frame suara kucing"):
        frame_kucing = y_panjang[:frame_len]; fft_kucing = np.fft.fft(frame_kucing)
        mag_kucing = np.abs(fft_kucing[:len(fft_kucing) // 2])
        freqs_kucing = np.fft.fftfreq(len(frame_kucing), 1 / sr)[:len(frame_kucing) // 2]
        fig8, ax8 = plt.subplots(figsize=(9, 2.5))
        ax8.plot(freqs_kucing, mag_kucing, color='#FF6B35', linewidth=1)
        ax8.fill_between(freqs_kucing, mag_kucing, alpha=0.08, color='#FF6B35')
        ax8.set_xlim(0, 4000); ax8.set_xlabel("Frekuensi (Hz)"); ax8.set_ylabel("Magnitudo")
        ax8.set_title("Spektrum frekuensi satu frame suara kucing"); ax8.grid(True)
        fig8.tight_layout(); st.pyplot(fig8); plt.close(fig8)
        st.caption("Banyak puncak frekuensi — suara kucing memang kompleks.")

# ── LANGKAH 5: SPEKTROGRAM ────────────────────────────────
st.subheader("5 — Spektrogram")
st.markdown('<span class="badge">Waktu + Frekuensi</span><span class="badge purple">2D Heat Map</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">Spektrogram</strong> = FFT banyak frame. Sumbu X = waktu, Y = frekuensi, warna = kekuatan.</p>', unsafe_allow_html=True)

t_gliss = np.linspace(0, 1, 16000); f_gliss = np.linspace(200, 1000, len(t_gliss))
y_gliss = np.sin(2 * np.pi * f_gliss * t_gliss)
D_gliss = librosa.amplitude_to_db(np.abs(librosa.stft(y_gliss)), ref=np.max)
fig9, ax9 = plt.subplots(figsize=(9, 3))
img = librosa.display.specshow(D_gliss, sr=16000, x_axis='time', y_axis='hz', ax=ax9, cmap='magma')
ax9.set_title("Spektrogram glissando (frekuensi naik 200 → 1000 Hz)")
plt.colorbar(img, ax=ax9, format="%+2.0f dB"); fig9.tight_layout(); st.pyplot(fig9); plt.close(fig9)
st.caption("Garis miring dari frekuensi rendah ke tinggi — itulah glissando.")

if cat_available:
    with st.expander("🐱 Spektrogram suara kucing (2 detik pertama)"):
        D_cat = librosa.amplitude_to_db(np.abs(librosa.stft(y_panjang)), ref=np.max)
        fig10, ax10 = plt.subplots(figsize=(9, 3))
        img = librosa.display.specshow(D_cat, sr=sr, x_axis='time', y_axis='hz', ax=ax10, cmap='magma')
        ax10.set_title("Spektrogram Cat.mp3")
        plt.colorbar(img, ax=ax10, format="%+2.0f dB"); fig10.tight_layout(); st.pyplot(fig10); plt.close(fig10)
        st.caption("Distribusi frekuensi berubah seiring waktu — kompleks dan dinamis.")

# ── LANGKAH 6: MFCC + DTW ────────────────────────────────
st.subheader("6 — MFCC + DTW (Dynamic Time Warping)")
st.markdown('<span class="badge">Feature Extraction</span><span class="badge blue">13 Koefisien</span><span class="badge purple">DTW Distance</span>', unsafe_allow_html=True)
st.markdown('<p style="color:#556070; font-size:0.9rem; line-height:1.8"><strong style="color:#00A050">MFCC</strong> meniru persepsi pendengaran manusia. <strong style="color:#00A050">DTW</strong> membandingkan dua sekuens MFCC dengan toleransi perbedaan durasi.</p>', unsafe_allow_html=True)

mfcc_gliss = librosa.feature.mfcc(y=y_gliss, sr=16000, n_mfcc=13)
fig11, ax11 = plt.subplots(figsize=(9, 3))
img = librosa.display.specshow(mfcc_gliss, sr=16000, x_axis='time', ax=ax11, cmap='viridis')
ax11.set_title("MFCC (13 koefisien) dari glissando")
plt.colorbar(img, ax=ax11); fig11.tight_layout(); st.pyplot(fig11); plt.close(fig11)
st.caption("Setiap baris = satu koefisien MFCC yang menunjukkan aspek berbeda dari spektrum.")

if cat_available:
    with st.expander("🐱 MFCC dari suara kucing (2 detik pertama)"):
        mfcc_cat = librosa.feature.mfcc(y=y_panjang, sr=sr, n_mfcc=13)
        fig12, ax12 = plt.subplots(figsize=(9, 3))
        img = librosa.display.specshow(mfcc_cat, sr=sr, x_axis='time', ax=ax12, cmap='viridis')
        ax12.set_title("MFCC suara kucing")
        plt.colorbar(img, ax=ax12); fig12.tight_layout(); st.pyplot(fig12); plt.close(fig12)
        st.caption("MFCC + DTW membandingkan pola ini antar rekaman secara akustik murni.")

# ============================================================
# ========== BAB 03: VOICE SIMILARITY CHALLENGE ==============
# ============================================================
st.markdown('<div class="section-pill" style="margin-top:2.5rem">🏆 Bab 03</div>', unsafe_allow_html=True)
st.header("Pengenalan Suara — Voice Similarity Challenge")

st.markdown("""
<div style="background:#F0FFF8; border:1px solid rgba(0,180,80,0.15); border-radius:14px; padding:1.2rem 1.6rem; margin-bottom:1.5rem">
<p style="color:#556070; margin:0; font-size:0.92rem; line-height:1.8">
Sistem menggunakan <strong style="color:#00A050">MFCC + DTW (Dynamic Time Warping)</strong> untuk mengukur
kemiripan akustik dua suara. Suara manusia normal → skor <strong style="color:#C03030">rendah</strong>,
tiruan kucing yang bagus → skor <strong style="color:#00A050">tinggi</strong>.
Skor: <strong>0 = sangat berbeda, 1 = identik</strong>.
</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📐 Mengapa MFCC + DTW lebih baik dari Cosine Similarity Resemblyzer?"):
    st.markdown("""
**Masalah Resemblyzer (GE2E Speaker Embeddings):**
- Dilatih untuk membedakan *identitas pembicara manusia*, bukan kemiripan akustik antar spesies
- Semua embedding manusia saling berdekatan (cosine ~0.75–0.95), rescaling `(sim+1)/2` memperparah inflasi skor
- Hasilnya: suara manusia biasa pun mendapat skor 0.8+ vs suara kucing

**Solusi MFCC + DTW:**
- MFCC mengukur *karakteristik akustik spektral* langsung dari sinyal
- DTW menyelaraskan dua sekuens dengan panjang berbeda secara optimal
- Cosine distance per frame mengukur perbedaan bentuk spektrum nyata
- Hasilnya: suara manusia normal → skor rendah (~0.10–0.25), tiruan kucing bagus → skor tinggi (~0.45–0.70)
    """)

# ── LOAD REFERENSI ────────────────────────────────────────
cat1_path = "Cat.mp3"
cat2_path = "Cat2.mp3"

if not os.path.exists(cat1_path):
    st.error(f"File {cat1_path} tidak ditemukan. Letakkan Cat.mp3 di folder yang sama.")
    st.stop()

feat_ref, y_ref_full, sr_ref = get_mfcc_features(cat1_path)
if feat_ref is None:
    st.error("Gagal memproses Cat.mp3.")
    st.stop()

st.markdown("**🎵 Suara Referensi**")
st.audio(cat1_path, format="audio/mpeg")
st.caption("Cat.mp3 — dijadikan patokan perbandingan akustik.")

# ── CAT1 VS CAT2 ─────────────────────────────────────────
cat2_available = os.path.exists(cat2_path)
if cat2_available:
    st.subheader("Cat.mp3 vs Cat2.mp3")
    feat_cat2, y_cat2, sr_cat2 = get_mfcc_features(cat2_path)
    if feat_cat2 is not None:
        sim_cat2, dtw_cat2 = compute_similarity_full(feat_ref, feat_cat2)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.audio(cat2_path, format="audio/mpeg")
        with col2:
            emoji_c2, label_c2, color_c2 = similarity_label(sim_cat2)
            bar_c2 = int(sim_cat2 * 100)
            st.markdown(f"""
            <div class="sim-card">
                <div class="sim-label">Kemiripan Akustik (MFCC + DTW)</div>
                <div class="sim-score">{sim_cat2:.3f}</div>
                <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{bar_c2}%"></div></div>
            </div>
            <span style="color:{color_c2}; font-size:0.9rem">{emoji_c2} {label_c2}</span>
            """, unsafe_allow_html=True)
        # Detail metrik Cat2
        render_metric_detail("Cat2.mp3", dtw_cat2, feat_ref, feat_cat2, y_ref_full, y_cat2, sr_ref, sim_cat2)
else:
    st.info("File Cat2.mp3 tidak tersedia. Bagian ini dilewati.")

# ── TANTANGAN ─────────────────────────────────────────────
st.subheader("🎙️ Tantangan: Siapa yang Lebih Mirip Kucing?")
st.markdown("""
<div style="background:#F0FFF8; border:1px solid rgba(0,180,80,0.12); border-radius:12px; padding:1rem 1.4rem; margin-bottom:1.2rem">
<p style="color:#556070; margin:0; font-size:0.88rem; line-height:1.7">
Rekam dua suara berbeda. Sistem membandingkan dengan <strong style="color:#00A050">Cat.mp3</strong>
menggunakan <strong style="color:#00A050">MFCC + DTW</strong>.
Setelah rekam, klik <em>"Detail Perhitungan Metrik"</em> di bawah kartu skor untuk melihat semua angka dan visualisasi.
</p>
</div>
""", unsafe_allow_html=True)

# session state
for key in ['human1_audio', 'human2_audio', 'human1_processed', 'human2_processed',
            'human1_dtw', 'human2_dtw', 'human1_y', 'human2_y']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ['human1_processed', 'human2_processed'] else False

col_h1, col_h2 = st.columns(2)

# ── MANUSIA 1 ─────────────────────────────────────────────
with col_h1:
    st.markdown("**👤 Manusia 1**")
    audio1 = st.audio_input("Rekam suara tiruan kucing (Manusia 1)", key="human1_input")
    if audio1 is not None:
        temp1 = "temp_human1.wav"
        with open(temp1, "wb") as f:
            f.write(audio1.getbuffer())
        feat1, y1, sr1 = get_mfcc_features(temp1)
        if feat1 is not None:
            sim1, dtw1 = compute_similarity_full(feat_ref, feat1)
            st.session_state.human1_audio     = (temp1, feat1, sim1)
            st.session_state.human1_dtw       = dtw1
            st.session_state.human1_y         = (feat1, y1)
            st.session_state.human1_processed = True
        else:
            if os.path.exists(temp1): os.remove(temp1)

    if st.session_state.human1_processed and st.session_state.human1_audio:
        _, feat1_s, sim1 = st.session_state.human1_audio
        bar1 = int(sim1 * 100)
        emoji1, label1, color1 = similarity_label(sim1)
        st.markdown(f"""
        <div class="sim-card">
            <div class="sim-label">Skor vs Cat.mp3 (MFCC + DTW)</div>
            <div class="sim-score">{sim1:.3f}</div>
            <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{bar1}%"></div></div>
        </div>
        <span style="color:{color1}; font-size:0.85rem">{emoji1} {label1}</span>
        """, unsafe_allow_html=True)
        # Detail metrik
        render_metric_detail(
            "Manusia 1",
            st.session_state.human1_dtw,
            feat_ref,
            st.session_state.human1_y[0],
            y_ref_full,
            st.session_state.human1_y[1],
            sr_ref,
            sim1,
        )

# ── MANUSIA 2 ─────────────────────────────────────────────
with col_h2:
    st.markdown("**👤 Manusia 2**")
    audio2 = st.audio_input("Rekam suara tiruan kucing (Manusia 2)", key="human2_input")
    if audio2 is not None:
        temp2 = "temp_human2.wav"
        with open(temp2, "wb") as f:
            f.write(audio2.getbuffer())
        feat2, y2, sr2 = get_mfcc_features(temp2)
        if feat2 is not None:
            sim2, dtw2 = compute_similarity_full(feat_ref, feat2)
            st.session_state.human2_audio     = (temp2, feat2, sim2)
            st.session_state.human2_dtw       = dtw2
            st.session_state.human2_y         = (feat2, y2)
            st.session_state.human2_processed = True
        else:
            if os.path.exists(temp2): os.remove(temp2)

    if st.session_state.human2_processed and st.session_state.human2_audio:
        _, feat2_s, sim2 = st.session_state.human2_audio
        bar2 = int(sim2 * 100)
        emoji2, label2, color2 = similarity_label(sim2)
        st.markdown(f"""
        <div class="sim-card">
            <div class="sim-label">Skor vs Cat.mp3 (MFCC + DTW)</div>
            <div class="sim-score">{sim2:.3f}</div>
            <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{bar2}%"></div></div>
        </div>
        <span style="color:{color2}; font-size:0.85rem">{emoji2} {label2}</span>
        """, unsafe_allow_html=True)
        render_metric_detail(
            "Manusia 2",
            st.session_state.human2_dtw,
            feat_ref,
            st.session_state.human2_y[0],
            y_ref_full,
            st.session_state.human2_y[1],
            sr_ref,
            sim2,
        )

# ── PEMENANG ──────────────────────────────────────────────
if st.session_state.human1_processed and st.session_state.human2_processed:
    if st.session_state.human1_audio and st.session_state.human2_audio:
        _, _, s1 = st.session_state.human1_audio
        _, _, s2 = st.session_state.human2_audio
        st.markdown("<br>", unsafe_allow_html=True)
        if abs(s1 - s2) < 0.02:
            winner_html = """<div style="background:linear-gradient(135deg,#FFF8E0,#FFFBE8);border:1px solid #C08000;border-radius:14px;padding:1.2rem 1.6rem;text-align:center"><div style="font-size:1.6rem">🤝</div><div style="font-size:1.1rem;font-weight:700;color:#7A5000;margin-top:0.4rem">Seri! Keduanya hampir sama miripnya.</div></div>"""
        elif s1 > s2:
            winner_html = f"""<div style="background:linear-gradient(135deg,#F0FFF8,#E0FFE8);border:1px solid #00C060;border-radius:14px;padding:1.2rem 1.6rem;text-align:center"><div style="font-size:1.6rem">🏆</div><div style="font-size:1.1rem;font-weight:700;color:#006030;margin-top:0.4rem">Manusia 1 menang! Skor {s1:.3f} vs {s2:.3f}</div><div style="font-size:0.8rem;color:#556070;margin-top:0.3rem">Suara Manusia 1 lebih mirip kucing secara akustik.</div></div>"""
        else:
            winner_html = f"""<div style="background:linear-gradient(135deg,#F0FFF8,#E0FFE8);border:1px solid #00C060;border-radius:14px;padding:1.2rem 1.6rem;text-align:center"><div style="font-size:1.6rem">🏆</div><div style="font-size:1.1rem;font-weight:700;color:#006030;margin-top:0.4rem">Manusia 2 menang! Skor {s2:.3f} vs {s1:.3f}</div><div style="font-size:0.8rem;color:#556070;margin-top:0.3rem">Suara Manusia 2 lebih mirip kucing secara akustik.</div></div>"""
        st.markdown(winner_html, unsafe_allow_html=True)

# ── RESET ─────────────────────────────────────────────────
if st.session_state.human1_processed or st.session_state.human2_processed:
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("🔄 Reset Rekaman", key="reset_human", use_container_width=True):
            for key in ['human1_audio', 'human2_audio']:
                if st.session_state[key]:
                    try: os.remove(st.session_state[key][0])
                    except: pass
            for key in ['human1_audio','human2_audio','human1_dtw','human2_dtw','human1_y','human2_y']:
                st.session_state[key] = None
            st.session_state.human1_processed = False
            st.session_state.human2_processed = False
            st.rerun()

# ============================================================
# ========== EXPLORE SECTION =================================
# ============================================================
st.markdown("---")
st.markdown('<div class="section-pill">🧪 Explore</div>', unsafe_allow_html=True)
st.header("Jelajahi Fitur Voice Lab")
st.markdown('<p style="color:#556070; font-size:0.9rem; margin-bottom:1.2rem">Setelah paham dasarnya, coba fitur-fitur seru berikut!</p>', unsafe_allow_html=True)

colA, colB, colC = st.columns(3)
with colA:
    st.page_link("pages/1_Meme_Challenge.py", label="🤣 Meme Voice Challenge", help="Tonton video meme, rekam suaramu, dan bandingkan kemiripannya!", use_container_width=True)
    st.caption("✅ Sudah tersedia")
with colB:
    st.page_link("pages/2_Classification.py", label="📊 Klasifikasi Suara 3D", help="Visualisasi dataset ESC-50 dalam ruang 3D, klik titik untuk mendengar suara.", use_container_width=True)
    st.caption("🆓 Eksplorasi 50 kelas suara")
with colC:
    st.button("🏎️ F1 Qualifying Voice (Segera)", disabled=True, use_container_width=True)
    st.caption("🚧 Segera hadir")

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0 0.5rem">
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#AABBCC; letter-spacing:3px; text-transform:uppercase">
        Voice Lab — Audio Processing Playground · MFCC + DTW Engine
    </span>
</div>
""", unsafe_allow_html=True)
