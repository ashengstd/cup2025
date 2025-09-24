
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from rich.logging import RichHandler
from rich.progress import track
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# This script re-uses logic from preprocess/base_data.py to generate
# visualizations without modifying the original files.

# Assuming features.py is in the python path
from features import (
    frequency_domain_features,
    time_domain_features,
    wavelet_packet_features,
)

# ========== Logger Setup ==========_=
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("feature_visualization")

# ========== Configuration ==========_=
SOURCE_DATA_DIR = "./数据集/源域数据集/12kHz_DE_data/"
SAMPLE_LENGTH = 1024  # Length of each smaller data sample
LABEL_MAP = {
    "B": "Ball",
    "IR": "Inner Race",
    # "N": "Normal", # Normal data is in a different folder, so we skip for this plot
    "OR": "Outer Race",
}


def process_all_files(
    root_dir: str, sample_length: int
) -> tuple[pd.DataFrame, np.ndarray]:
    all_file_paths = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.endswith(".mat")
    ]

    all_features_list = []
    all_labels_list = []

    # Filter out paths that don't have a valid label key in their path
    valid_paths = [
        p for p in all_file_paths if any(f"/{key}/" in p for key in LABEL_MAP)
    ]

    for file_path in track(valid_paths, description="[bold cyan]Processing Files"):
        mat_data = sio.loadmat(file_path)

        label_key = "Unknown"
        for key in LABEL_MAP:
            if f"/{key}/" in file_path:
                label_key = key
                break
        label = LABEL_MAP.get(label_key, "Unknown")

        de_key = next((k for k in mat_data if "DE" in k), None)
        if not de_key:
            logger.warning(f"DE signal not found in {file_path}, skipping.")
            continue

        signal = mat_data[de_key].flatten()
        num_samples = len(signal) // sample_length
        if num_samples == 0:
            continue

        segments = signal[: num_samples * sample_length].reshape(
            num_samples, sample_length
        )
        segments_scaled = StandardScaler().fit_transform(segments)

        file_features = {}
        file_features.update(time_domain_features(segments_scaled))
        file_features.update(frequency_domain_features(segments_scaled, fs=12000))
        file_features.update(wavelet_packet_features(segments_scaled))

        all_features_list.append(pd.DataFrame(file_features))
        all_labels_list.extend([label] * num_samples)

    if not all_features_list:
        raise ValueError(
            "No data was processed. Check directory path and file structure."
        )

    final_features_df = pd.concat(all_features_list, ignore_index=True)
    final_labels_array = np.array(all_labels_list)

    return final_features_df, final_labels_array


# ========== Main Execution ==========_=
logger.info("Starting feature extraction for visualization...")
# We only process the 12kHz data which contains fault types
features_df, labels = process_all_files(SOURCE_DATA_DIR, SAMPLE_LENGTH)

logger.info(f"Total Samples: {features_df.shape[0]}")
logger.info(f"Features Shape: {features_df.shape}")
logger.info(f"Unique Labels: {', '.join(np.unique(labels))}")

# ========== Scaling and Dimensionality Reduction ==========_=
# Drop columns with zero variance, which can cause issues with PCA/t-SNE
features_df = features_df.loc[:, features_df.var() > 1e-6]

X_scaled = StandardScaler().fit_transform(features_df.values)
label_codes, unique_labels = pd.factorize(labels)

logger.info("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

logger.info("Running t-SNE... (this may take a moment)")
# Use a smaller subset for t-SNE if the dataset is large to speed it up
num_points = min(5000, X_scaled.shape[0])
subset_indices = np.random.choice(X_scaled.shape[0], num_points, replace=False)

tsne = TSNE(
    n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto"
)
X_tsne = tsne.fit_transform(X_scaled[subset_indices, :])

# ========== Plotting ==========_=
logger.info("Generating plots...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Feature Space Visualization of Bearing Faults", fontsize=20)
scatter_params = {"s": 10, "alpha": 0.7}

# --- PCA Plot ---
scatter_pca = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=label_codes, cmap="viridis", **scatter_params
)
axes[0].set_title("PCA Visualization", fontsize=16)
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(handles=scatter_pca.legend_elements()[0], labels=list(unique_labels))

# --- t-SNE Plot ---
scatter_tsne = axes[1].scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=label_codes[subset_indices],
    cmap="viridis",
    **scatter_params,
)
axes[1].set_title("t-SNE Visualization (on 5000 samples)", fontsize=16)
axes[1].set_xlabel("t-SNE Dimension 1")
axes[1].set_ylabel("t-SNE Dimension 2")
axes[1].legend(handles=scatter_tsne.legend_elements()[0], labels=list(unique_labels))

# --- Save and Show ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = "./plots/feature_visualization.svg"
plt.savefig(output_path, format="svg")
logger.info(f"Feature visualization plot saved to {output_path}")
plt.show()
