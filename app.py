# ========== SPECTRAL DISTANCE FUNCTIONS (FIXED) ==========

def extract_spectral_features(file_path, sr_target=16000, n_mfcc=13):
    """
    Ekstrak fitur spektral dan pitch untuk membedakan suara.
    Penambahan Pitch (F0) dan Spectral Flatness sangat krusial untuk 
    membedakan suara kucing (pitch tinggi, sangat tonal) dengan manusia.
    """
    y, sr = librosa.load(file_path, sr=sr_target)
    
    # Batasi maksimal 10 detik untuk ekstraksi fitur agar performa Streamlit tetap cepat
    max_samples = sr_target * 10
    if len(y) > max_samples:
        y = y[:max_samples]
    
    # 1. Pitch (Fundamental Frequency) - Pembeda utama Kucing vs Manusia
    # fmin=80, fmax=2000 mencakup rentang suara manusia dan kucing
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=2000, sr=sr)
    f0_mean = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
    f0_std  = float(np.nanstd(f0))  if not np.all(np.isnan(f0)) else 0.0
    
    # 2. Spectral Features
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    flatness  = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # 3. MFCCs (ambil 5 pertama untuk merepresentasikan timbre)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means= np.mean(mfcc, axis=1)[:5]
    
    # Gabungkan menjadi satu vektor (Total 12 dimensi)
    feature_vector = np.concatenate([
        [f0_mean, f0_std],
        [centroid, rolloff, bandwidth, flatness, zcr],
        mfcc_means
    ]).astype(float)
    
    return feature_vector, y, sr


def spectral_distance(vec_a, vec_b):
    """
    Hitung jarak menggunakan Normalized Euclidean Distance.
    
    Kenapa bukan Cosine Distance?
    Cosine hanya melihat "sudut" (proporsi relatif). Karena manusia dan kucing 
    sama-sama mamalia yang bersuara dari pita suara, bentuk spektrum (MFCC) mereka 
    secara proporsi memang mirip, sehingga Cosine memberi skor tinggi.
    
    Euclidean Distance dengan normalisasi akan menghukum perbedaan nilai absolut 
    (terutama pada Pitch/F0 yang sangat berbeda: Kucing ~600Hz, Manusia ~150Hz).
    """
    # Skala normalisasi untuk setiap fitur (berdasarkan rentang tipikal)
    # Urutan: f0_mean, f0_std, centroid, rolloff, bandwidth, flatness, zcr, mfcc0..4
    SCALES = np.array([
        250.0,  # f0_mean (Kucing ~600Hz, Manusia ~150Hz -> beda ~450)
        60.0,   # f0_std
        600.0,  # centroid
        1000.0, # rolloff
        500.0,  # bandwidth
        0.05,   # flatness (Kucing sangat tonal/rendah, manusia lebih noisy/tinggi)
        0.05,   # zcr
        150.0, 100.0, 80.0, 80.0, 80.0  # MFCCs 0-4
    ])
    
    # Pastikan vektor memiliki panjang yang sama dengan SCALES (jaga-jaga jika ada perubahan)
    min_len = min(len(vec_a), len(vec_b), len(SCALES))
    vec_a = vec_a[:min_len]
    vec_b = vec_b[:min_len]
    scales = SCALES[:min_len]
    
    # Hitung Normalized Euclidean Distance
    diff = (vec_a - vec_b) / scales
    dist = np.sqrt(np.sum(diff ** 2))
    
    # Konversi distance ke similarity score [0, 1]
    # Menggunakan fungsi eksponensial: exp(-dist / sigma)
    # sigma = 2.5 menentukan seberapa ketat penal jaraknya
    sigma = 2.5
    similarity = float(np.exp(-dist / sigma))
    
    return similarity, float(dist)
