import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fft import fft

# --- Configuration ---
# Using multiple files per category to see variation
SAMPLE_FILES = {
    "Normal": {
        "paths": [
            "./数据集/源域数据集/48kHz_Normal_data/N_1_(1772rpm).mat",
            "./数据集/源域数据集/48kHz_Normal_data/N_2_(1750rpm).mat",
        ],
        "sr": 48000,
    },
    "Inner Race": {
        "paths": [
            "./数据集/源域数据集/12kHz_DE_data/IR/0028/IR028_0_(1797rpm).mat",
            "./数据集/源域数据集/12kHz_DE_data/IR/0028/IR028_2_(1750rpm).mat",
        ],
        "sr": 12000,
    },
    "Ball": {
        "paths": [
            "./数据集/源域数据集/12kHz_DE_data/B/0028/B028_0_(1797rpm).mat",
            "./数据集/源域数据集/12kHz_DE_data/B/0028/B028_2_(1750rpm).mat",
        ],
        "sr": 12000,
    },
    "Outer Race": {
        "paths": [
            "./数据集/源域数据集/12kHz_DE_data/OR/Opposite/0021/OR021@12_2.mat",
            "./数据集/源域数据集/12kHz_DE_data/OR/Opposite/0021/OR021@12_3.mat",
        ],
        "sr": 12000,
    },
}
# Increased sample length for more detail
N_SAMPLES = 8192
OUTPUT_DIR = "./plots/frequency_analysis_svgs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Main Processing Loop ---
for label, info in SAMPLE_FILES.items():
    sampling_rate = info["sr"]
    for i, full_path in enumerate(info["paths"]):
        print(f"Processing {label} sample {i+1}: {full_path}")

        if not os.path.exists(full_path):
            print(f"  -> File not found. Skipping.")
            continue

        # Load data
        mat_data = sio.loadmat(full_path)
        data_key = next((k for k in mat_data if "DE_time" in k), None)
        if not data_key:
            valid_keys = [
                k
                for k in mat_data
                if isinstance(mat_data[k], np.ndarray) and not k.startswith("__")
            ]
            if valid_keys:
                data_key = max(valid_keys, key=lambda k: mat_data[k].size)

        if not data_key:
            print(f"  -> No valid signal found. Skipping.")
            continue

        signal = mat_data[data_key].flatten()

        if len(signal) < N_SAMPLES:
            print(f"  -> Signal is too short. Skipping.")
            continue

        signal_segment = signal[:N_SAMPLES]

        # --- Create a new figure for each sample ---
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Analysis for {label} - Sample {i+1}", fontsize=16)

        # --- Time Domain Plot ---
        time_ax = axes[0]
        time_vector = np.arange(N_SAMPLES) / sampling_rate
        time_ax.plot(time_vector, signal_segment, lw=1)
        time_ax.set_xlabel("Time (s)")
        time_ax.set_ylabel("Amplitude")
        time_ax.set_title("Time Domain")
        time_ax.grid(True)

        # --- Frequency Domain Plot (FFT) ---
        freq_ax = axes[1]
        fft_vals = fft(signal_segment)
        fft_freq = np.fft.fftfreq(N_SAMPLES, 1 / sampling_rate)

        positive_mask = fft_freq >= 0
        freq_ax.plot(
            fft_freq[positive_mask], np.abs(fft_vals[positive_mask]), lw=1.5
        )
        freq_ax.set_xlabel("Frequency (Hz)")
        freq_ax.set_ylabel("Magnitude")
        freq_ax.set_title("Frequency Domain (FFT)")
        freq_ax.set_xlim(0, sampling_rate / 2)
        freq_ax.grid(True)

        # --- Save figure as SVG ---
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        output_filename = f"{label.replace(' ', '_')}_{i+1}.svg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, format="svg")
        print(f"  -> Saved plot to {output_path}")
        plt.close(fig)  # Close the figure to free memory

print("\nDone. All plots saved in SVG format.")
