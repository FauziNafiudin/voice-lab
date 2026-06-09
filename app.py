@st.cache_data
def extract_spectral_features(file_path, sr_target=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr_target)

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc, axis=1)

    centroid_m  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / (sr / 2)
    rolloff_m   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)) / (sr / 2)
    bandwidth_m = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / (sr / 2)

    contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_m  = np.mean(contrast, axis=1)
    contrast_n  = contrast_m / (np.max(np.abs(contrast_m)) + 1e-9)

    zcr_m       = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness_m  = np.mean(librosa.feature.spectral_flatness(y=y))

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

    if len(f0_voiced) > 0:
        f0_mean      = np.mean(f0_voiced) / 2000.0
        f0_std       = np.std(f0_voiced) / 1000.0
        f0_range     = (np.max(f0_voiced) - np.min(f0_voiced)) / 2000.0
        voiced_ratio = len(f0_voiced) / (len(f0) + 1e-9)
    else:
        f0_mean = f0_std = f0_range = voiced_ratio = 0.0

    y_harm, y_perc = librosa.effects.hpss(y)
    hnr_norm = np.clip(
        np.mean(y_harm ** 2) / (np.mean(y_perc ** 2) + 1e-9) / 10.0,
        0, 1
    )

    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std * 0.5,
        [centroid_m],
        [rolloff_m],
        [bandwidth_m],
        contrast_n,
        [zcr_m],
        [flatness_m],
        [f0_mean    * 3.0],
        [f0_std     * 2.0],
        [f0_range   * 2.0],
        [voiced_ratio],
        [hnr_norm   * 2.0],
    ])
    return feature_vector, y, sr


def spectral_distance(vec_a, vec_b):
    from scipy.spatial.distance import cosine, euclidean

    cos_sim = 1.0 - (cosine(vec_a, vec_b) / 2.0)
    euc_sim = 1.0 - np.tanh(euclidean(vec_a, vec_b) / 5.0)
    similarity = 0.35 * cos_sim + 0.65 * euc_sim

    return float(similarity), float(cosine(vec_a, vec_b))
