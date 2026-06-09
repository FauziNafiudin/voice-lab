# ========== FUNGSI EKSTRAKSI FITUR — VERSI BARU ==========
@st.cache_data
def extract_spectral_features(file_path, sr_target=16000, n_mfcc=13):
    """
    Ekstrak fitur spektral + bioacoustic dari file audio.
    Fitur baru vs lama:
      + Pitch (F0) mean & range   → pembeda spesies terkuat (kucing ~600-1500 Hz, manusia ~80-300 Hz)
      + Harmonic-to-Noise Ratio   → kucing lebih harmonic, manusia lebih noisy
      + Spectral Flatness         → kucing punya puncak tajam, manusia lebih flat
      + Pitched frame ratio       → proporsi frame yang benar-benar bersuara
    """
    y, sr = librosa.load(file_path, sr=sr_target)

    # --- MFCC (timbre/spectral envelope) ---
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc, axis=1)  # tambah std untuk variabilitas

    # --- Spectral features dasar ---
    centroid    = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_m  = np.mean(centroid) / (sr / 2)  # normalisasi ke [0,1]

    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_m   = np.mean(rolloff) / (sr / 2)

    bandwidth   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_m = np.mean(bandwidth) / (sr / 2)

    contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_m  = np.mean(contrast, axis=1)
    contrast_n  = contrast_m / (np.max(np.abs(contrast_m)) + 1e-9)

    zcr         = librosa.feature.zero_crossing_rate(y)
    zcr_m       = np.mean(zcr)

    # --- Spectral Flatness (baru) ---
    # Mendekati 1 = noise/flat; mendekati 0 = ada nada/harmonic
    flatness    = librosa.feature.spectral_flatness(y=y)
    flatness_m  = np.mean(flatness)

    # --- Pitch / F0 (baru) — pembeda spesies terkuat ---
    # Kucing: F0 ~600-1500 Hz | Manusia: ~80-300 Hz
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),   # ~65 Hz — batas bawah
        fmax=librosa.note_to_hz('C7'),   # ~2093 Hz — batas atas (menangkap kucing)
        sr=sr
    )
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

    if len(f0_voiced) > 0:
        f0_mean  = np.mean(f0_voiced) / 2000.0    # normalisasi ke ~[0,1]
        f0_std   = np.std(f0_voiced) / 1000.0
        f0_range = (np.max(f0_voiced) - np.min(f0_voiced)) / 2000.0
        voiced_ratio = len(f0_voiced) / (len(f0) + 1e-9)  # proporsi frame bersuara
    else:
        f0_mean  = 0.0
        f0_std   = 0.0
        f0_range = 0.0
        voiced_ratio = 0.0

    # --- Harmonic-to-Noise Ratio proxy (baru) ---
    # Pisahkan komponen harmonic dan percussive/noise
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.mean(y_harm ** 2)
    perc_energy = np.mean(y_perc ** 2)
    hnr_proxy   = harm_energy / (perc_energy + 1e-9)
    hnr_norm    = np.clip(hnr_proxy / 10.0, 0, 1)  # normalisasi ke [0,1]

    # --- Gabungkan semua fitur dengan bobot eksplisit ---
    # Catatan: MFCC std dan fitur bioacoustic ditambahkan sebagai dimensi baru
    feature_vector = np.concatenate([
        mfcc_mean,                        # 13 dim — timbre
        mfcc_std * 0.5,                   # 13 dim — variabilitas timbre (bobot lebih kecil)
        [centroid_m],                     # 1 dim
        [rolloff_m],                      # 1 dim
        [bandwidth_m],                    # 1 dim
        contrast_n,                       # 7 dim
        [zcr_m],                          # 1 dim
        [flatness_m],                     # 1 dim — baru
        [f0_mean   * 3.0],               # 1 dim — BOBOT ×3: pembeda spesies terkuat
        [f0_std    * 2.0],               # 1 dim — BOBOT ×2
        [f0_range  * 2.0],               # 1 dim — BOBOT ×2
        [voiced_ratio],                   # 1 dim — proporsi voiced frames
        [hnr_norm  * 2.0],               # 1 dim — BOBOT ×2: harmonic vs noise
    ])
    return feature_vector, y, sr


def spectral_distance(vec_a, vec_b):
    """
    Ensemble distance: Cosine + Weighted Euclidean
    
    Mengapa tidak hanya cosine?
    - Cosine hanya mengukur sudut vektor, mengabaikan magnitudo
    - Pitch kucing (F0 ~1000 Hz) vs manusia (~150 Hz) berbeda BESAR dalam magnitudo
    - Weighted Euclidean menangkap perbedaan absolut ini
    
    Hasil: similarity [0, 1], semakin tinggi = semakin mirip
    """
    from scipy.spatial.distance import cosine, euclidean

    # Cosine similarity
    cos_dist = cosine(vec_a, vec_b)           # [0, 2]
    cos_sim  = 1.0 - (cos_dist / 2.0)         # [0, 1]

    # Weighted Euclidean — normalisasi ke [0,1] via tanh
    euc_dist = euclidean(vec_a, vec_b)
    euc_sim  = 1.0 - np.tanh(euc_dist / 5.0)  # tanh agar bounded [0,1]

    # Ensemble: bobot lebih besar ke Euclidean karena menangkap perbedaan pitch absolut
    similarity = 0.35 * cos_sim + 0.65 * euc_sim

    return float(similarity), float(cos_dist)
