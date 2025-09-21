import numpy as np
import pywt
from scipy.fftpack import fft


def time_domain_features(data) -> dict:
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    rms = np.sqrt(np.mean(data**2, axis=1))
    skewness = np.mean(((data - mean[:, None]) / std[:, None]) ** 3, axis=1)
    kurtosis = np.mean(((data - mean[:, None]) / std[:, None]) ** 4, axis=1)
    peak_to_peak = np.ptp(data, axis=1)
    crest_factor = np.max(np.abs(data), axis=1) / rms
    impulse_factor = np.max(np.abs(data), axis=1) / mean
    margin_factor = np.max(np.abs(data), axis=1) / (
        np.mean(np.sqrt(np.abs(data)), axis=1) ** 2
    )
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor,
    }


def frequency_domain_features(data) -> dict:
    # f = np.fft.fftfreq(data.shape[1], d=1 / fs)
    fft_values = np.abs(fft(data, axis=1))
    power = np.mean(fft_values**2, axis=1)
    return {"power": power}


def wavelet_packet_features(data, wavelet="db4", maxlevel=3):
    features = []
    for i in range(data.shape[0]):
        wp = pywt.WaveletPacket(
            data=data[i, :], wavelet=wavelet, mode="symmetric", maxlevel=maxlevel
        )
        nodes = [node.path for node in wp.get_level(maxlevel, order="freq")]
        energies = [np.sum(wp[node].data ** 2) for node in nodes]
        energies = np.array(energies)
        energies_ratio = energies / np.sum(energies)
        features.append(energies_ratio)
    features = np.array(features)
    col_names = [f"wp_{i}" for i in range(features.shape[1])]
    return {name: features[:, idx] for idx, name in enumerate(col_names)}
