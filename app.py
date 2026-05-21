import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

import streamlit as st

st.set_page_config(
    page_title="Voice Lab",
    page_icon="🎤",
    layout="wide"
)

import streamlit as st

st.set_page_config(
    page_title="Voice Lab",
    page_icon="🎤",
    layout="wide"
)

# ------- CUSTOM CSS TEMA AUDIO -------
st.markdown("""
<style>
    /* Latar belakang utama - gelap */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161A22;
    }

    /* Judul & heading */
    h1, h2, h3, h4 {
        color: #00E676;  /* hijau neon audio */
    }

    /* Tombol umum */
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1ED760;
        box-shadow: 0 0 15px #1ED760;
    }

    /* Metric box */
    [data-testid="stMetric"] {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 10px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1A1C23;
        border: 1px solid #333;
    }

    /* Input file/audio */
    .stAudioInput>div {
        background-color: #1E1E2E;
        border-radius: 10px;
    }

    /* Caption */
    .caption {
        color: #888;
    }

    /* Link/page link */
    a {
        color: #1DB954 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🎤 Voice Lab – Laboratorium Suara")
st.write("""
Selamat datang! Di sini kita akan menjelajahi dunia **audio processing** dengan cara yang interaktif.
Kamu bisa merekam suara, melihat bentuk gelombangnya, mengenali siapa yang bicara, sampai menirukan suara meme.
Semua bisa kamu coba langsung di browser!
""")

# ---------- BAGIAN 1: APA ITU AUDIO? ----------
st.header("🔊 Apa Itu Audio?")
st.write("""
Audio adalah getaran udara yang ditangkap telinga dan direpresentasikan sebagai sinyal digital oleh komputer.
Setiap suara punya **bentuk gelombang** (waveform) yang unik. Misalnya, suara kucing mengeong punya pola tertentu.
""")

# Contoh waveform suara kucing (sintetis sederhana)
with st.expander("🐱 Lihat contoh waveform suara kucing"):
    # Generate sinyal dummy
    sr = 16000
    t = np.linspace(0, 1, sr)
    # Frekuensi mengeong: sekitar 400-800 Hz
    y_cat = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)
    
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t[:2000], y_cat[:2000])  # tampilkan 2000 sampel pertama
    ax.set_title("Waveform suara kucing (sintetis)")
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    st.pyplot(fig)
    st.caption("Bentuk gelombang di atas adalah simulasi sederhana dari suara 'meong'.")

# ---------- BAGIAN 2: VOICE RECOGNITION ----------
st.header("🗣️ Voice Recognition")
st.write("""
**Voice Recognition** (pengenalan suara) adalah teknologi yang memungkinkan komputer memahami suara manusia.
Contohnya:
- Mengubah ucapan menjadi teks (speech-to-text)
- Mengenali siapa yang berbicara (speaker recognition)
- Mendeteksi emosi dari nada bicara
""")

# Sneak peek interaktif: rekam dan lihat waveform-mu sendiri
st.subheader("🎙️ Sneak Peek: Rekam Suaramu & Lihat Waveform-nya")
st.write("Coba rekam sesuatu, misalnya suara kamu menirukan kucing atau sekadar bilang 'halo'.")

audio_value = st.audio_input("Klik ikon mikrofon untuk merekam")

if audio_value is not None:
    # Simpan ke file sementara
    with open("temp_user_audio.wav", "wb") as f:
        f.write(audio_value.getbuffer())
    
    # Baca dengan librosa
    y, sr = librosa.load("temp_user_audio.wav", sr=16000)
    
    # Plot waveform
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(y)
    ax.set_title("Waveform Suara Kamu")
    ax.set_xlabel("Sampel")
    ax.set_ylabel("Amplitudo")
    st.pyplot(fig)
    
    st.success("Ini dia bentuk gelombang suara kamu! Setiap suara punya pola berbeda.")
    
    # Hapus file sementara
    if os.path.exists("temp_user_audio.wav"):
        os.remove("temp_user_audio.wav")

# ---------- BAGIAN 3: JELAJAHI FITUR ----------
st.header("🧪 Jelajahi Fitur Voice Lab")
st.write("Setelah tahu dasar-dasarnya, sekarang saatnya bermain dengan fitur-fitur seru!")

col1, col2 = st.columns(2)

with col1:
    # Tombol menuju halaman Meme Challenge
    st.page_link(
        "pages/1_Meme_Challenge.py",
        label="🤣 Meme Voice Challenge",
        help="Tonton video meme, rekam suaramu, dan bandingkan kemiripannya!",
        use_container_width=True
    )
    st.caption("✅ Sudah tersedia")

with col2:
    st.button(
        "🏎️ F1 Qualifying Voice (Segera)",
        disabled=True,
        use_container_width=True
    )
    st.caption("🚧 Segera hadir")

# Footer
st.markdown("---")
st.caption("💡 Tambah fitur baru dengan membuat file Python di folder `pages/`, misal `2_F1_Qualifying.py`.")