import librosa
import numpy as np

def Normalize(data):
    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler(feature_range=(0, 1))  # 加载函数
    b = scalar.fit_transform(data)  # 归一化
    return b

def get_melspectrogram_flux(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    # y1 = y[0:2048]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    # S = librosa.feature.melspectrogram(y=y1, sr=sr,n_fft=2048, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
    # print(S.shape)
    S_dB = 10 * np.log10(1e-10 + S)
    # S_dB = Normalize(S_dB)
    mf = np.zeros((1,S_dB.shape[1]))
    half_p = int(S_dB.shape[0]*0.5)
    for i in range(S_dB.shape[1]):
        mf[:, i] = sum((S_dB[0:half_p, i]))
        # mf[:, i] = sum((S_dB[:, i]))
    sf_line = mf[0]
    import scipy.signal as signal
    b, a = signal.butter(3, 0.1, analog=False)
    sig_sf = signal.filtfilt(b, a, sf_line)
    return sf_line,sig_sf

def get_spectral_flux(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    duration = librosa.get_duration(y,sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=441, center=False))
    sf = np.zeros((1, S.shape[1] - 1))
    for i in range(S.shape[1] - 1):
        sf[:, i] = sum((S[:, i + 1] - S[:, i]) ** 2)
    sf_line = sf[0]
    import scipy.signal as signal
    b, a = signal.butter(3, 0.2, analog=False)
    sig_sf = signal.filtfilt(b, a, sf_line)
    return sf_line, sig_sf,duration

def get_spectral_flux_v2(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    duration = librosa.get_duration(y,sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=441, center=False))
    sf = np.zeros((1, S.shape[1] - 1))
    for i in range(S.shape[1] - 1):
        sf[:, i] = sum((S[:, i + 1] - S[:, i]))
    sf_line = sf[0]
    import scipy.signal as signal
    b, a = signal.butter(3, 0.2, analog=False)
    sig_sf = signal.filtfilt(b, a, sf_line)
    return sf_line, sig_sf,duration

def get_troughs_from_sig_sf(sig_sf):
    troughs = [i for i in range(len(sig_sf) - 1) if sig_sf[i - 1] > sig_sf[i] < sig_sf[i + 1] and sig_sf[i] < -10]
    return troughs

def get_peak_from_sig_sf(sig_sf):
    peaks = [i for i in range(len(sig_sf)-1) if sig_sf[i - 1] < sig_sf[i] > sig_sf[i + 1] and sig_sf[i] > 0.5]
    return peaks

def get_troughs_from_file(file_path):
    sf_line, sig_sf,duration = get_spectral_flux_v2(file_path)
    troughs = get_troughs_from_sig_sf(sig_sf)
    return troughs

def get_peaks_from_file(file_path):
    sf_line, sig_sf,duration = get_spectral_flux(file_path)
    peaks = get_peak_from_sig_sf(sig_sf)
    return peaks

def get_peaks_in_time_from_file(file_path):
    sf_line, sig_sf,duration = get_spectral_flux(file_path)
    peaks = get_peak_from_sig_sf(sig_sf)
    hop_time = 10e-3
    peaks_times = [p*hop_time for p in peaks]
    return peaks_times
