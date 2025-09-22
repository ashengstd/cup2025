import numpy as np
import pywt
from scipy.fftpack import fft
from scipy.signal import hilbert


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


def frequency_domain_features(data, fs: int) -> dict:
    fft_values = np.abs(fft(data, axis=1))
    power = np.mean(fft_values**2, axis=1)

    # --- New Features ---
    fft_freq = np.fft.fftfreq(data.shape[1], d=1.0 / fs)

    # Filter for positive frequencies only
    positive_freq_mask = fft_freq > 0
    if not np.any(positive_freq_mask):
        # Handle case with no positive frequencies
        zeros = np.zeros_like(power)
        return {
            "power": power,
            "freq_centroid": zeros,
            "freq_std": zeros,
            "freq_skew": zeros,
            "freq_kurtosis": zeros,
        }

    fft_values_pos = fft_values[:, positive_freq_mask]
    fft_freq_pos = fft_freq[positive_freq_mask]

    # Normalize spectrum for statistical calculations
    fft_norm = fft_values_pos / np.sum(fft_values_pos, axis=1, keepdims=True)

    # Frequency Centroid (Mean Frequency)
    freq_centroid = np.sum(fft_freq_pos * fft_norm, axis=1)

    # Frequency Standard Deviation
    freq_std = np.sqrt(
        np.sum(fft_norm * (fft_freq_pos - freq_centroid[:, None]) ** 2, axis=1)
    )

    # To avoid division by zero for signals with no frequency variation
    safe_std = np.where(freq_std == 0, 1, freq_std)

    # Frequency Skewness & Kurtosis
    freq_skew = np.sum(
        fft_norm * ((fft_freq_pos - freq_centroid[:, None]) / safe_std[:, None]) ** 3, axis=1
    )
    freq_kurtosis = np.sum(
        fft_norm * ((fft_freq_pos - freq_centroid[:, None]) / safe_std[:, None]) ** 4, axis=1
    )

    return {
        "power": power,
        "freq_centroid": freq_centroid,
        "freq_std": freq_std,
        "freq_skew": freq_skew,
        "freq_kurtosis": freq_kurtosis,
    }


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


def envelope_spectrum_features(data, fs: int) -> dict:
    # Hilbert transform to get the analytic signal
    analytic_signal = hilbert(data, axis=1)
    # Get the envelope
    envelope = np.abs(analytic_signal)
    # Remove DC component
    envelope = envelope - np.mean(envelope, axis=1, keepdims=True)

    # Now, treat the envelope as the new signal and get its frequency domain features
    fft_values = np.abs(fft(envelope, axis=1))
    power = np.mean(fft_values**2, axis=1)

    fft_freq = np.fft.fftfreq(data.shape[1], d=1.0 / fs)

    positive_freq_mask = fft_freq > 0
    if not np.any(positive_freq_mask):
        zeros = np.zeros_like(power)
        return {
            "env_power": power,
            "env_freq_centroid": zeros,
            "env_freq_std": zeros,
            "env_freq_skew": zeros,
            "env_freq_kurtosis": zeros,
        }

    fft_values_pos = fft_values[:, positive_freq_mask]
    fft_freq_pos = fft_freq[positive_freq_mask]

    fft_norm = fft_values_pos / np.sum(fft_values_pos, axis=1, keepdims=True)

    freq_centroid = np.sum(fft_freq_pos * fft_norm, axis=1)
    freq_std = np.sqrt(
        np.sum(fft_norm * (fft_freq_pos - freq_centroid[:, None]) ** 2, axis=1)
    )

    safe_std = np.where(freq_std == 0, 1, freq_std)

    freq_skew = np.sum(
        fft_norm * ((fft_freq_pos - freq_centroid[:, None]) / safe_std[:, None]) ** 3, axis=1
    )
    freq_kurtosis = np.sum(
        fft_norm * ((fft_freq_pos - freq_centroid[:, None]) / safe_std[:, None]) ** 4, axis=1
    )

    return {
        "env_power": power,
        "env_freq_centroid": freq_centroid,
        "env_freq_std": freq_std,
        "env_freq_skew": freq_skew,
        "env_freq_kurtosis": freq_kurtosis,
    }
