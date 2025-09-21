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

from features import (
    frequency_domain_features,
    time_domain_features,
    wavelet_packet_features,
)

# ========== Logger Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("feature_extraction")

SOURCE_DATA_DIR = "./数据集/源域数据集/12kHz_DE_data/"
SAMPLE_LENGTH = 1024  # Length of each smaller data sample
LABEL_MAP = {"B": "Ball", "IR": "Inner Race", "N": "Normal", "OR": "Outer Race"}


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

    for file_path in track(all_file_paths, description="[bold cyan]Processing Files"):
        mat_data = sio.loadmat(file_path)

        try:
            path_parts = file_path.split(os.sep)
            data_folder_index = path_parts.index("12kHz_DE_data")
            label_key = path_parts[data_folder_index + 1]
            label = LABEL_MAP.get(label_key, "Unknown")
        except ValueError:
            logger.warning(f"Skipping file with unexpected path: {file_path}")
            continue

        de_key = next((k for k in mat_data if "DE" in k), None)
        fe_key = next((k for k in mat_data if "FE" in k), None)
        ba_key = next((k for k in mat_data if "BA" in k), None)

        if not all([de_key, fe_key, ba_key]):
            logger.warning(f"Missing signals in file: {file_path}")
            continue

        signals = {
            "DE": mat_data[de_key].flatten(),
            "FE": mat_data[fe_key].flatten(),
            "BA": mat_data[ba_key].flatten(),
        }

        min_len = min(len(s) for s in signals.values())
        num_samples = min_len // sample_length
        if num_samples == 0:
            logger.warning(f"File too short for segmentation: {file_path}")
            continue

        file_features = {}
        for prefix, signal_data in signals.items():
            segments = signal_data[: num_samples * sample_length].reshape(
                num_samples, sample_length
            )
            segments_scaled = StandardScaler().fit_transform(segments)

            file_features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in time_domain_features(segments_scaled).items()
                }
            )
            file_features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in frequency_domain_features(segments_scaled).items()
                }
            )
            file_features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in wavelet_packet_features(segments_scaled).items()
                }
            )

        all_features_list.append(pd.DataFrame(file_features))
        all_labels_list.extend([label] * num_samples)

    if not all_features_list:
        raise ValueError(
            "No data was processed. Check your directory path and file structure."
        )

    final_features_df = pd.concat(all_features_list, ignore_index=True)
    final_labels_array = np.array(all_labels_list)

    return final_features_df, final_labels_array


# ========== Process Files ==========
features_df, labels = process_all_files(SOURCE_DATA_DIR, SAMPLE_LENGTH)

# ========== Summary ==========
logger.info(f"Total Samples: {features_df.shape[0]}")
logger.info(f"Features Shape: {features_df.shape}")
logger.info(f"Labels Shape: {labels.shape}")
logger.info(f"Unique Labels: {', '.join(np.unique(labels))}")

# ========== Visualization ==========
X_scaled = StandardScaler().fit_transform(features_df.values)
label_codes, unique_labels = pd.factorize(labels)

logger.info("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
np.savez("./data/pca_features_labels.npz", X_pca=X_pca, labels=labels)
logger.info("PCA features and labels saved to 'pca_features_labels.npz'.")

logger.info("Running t-SNE...")
tsne = TSNE(
    n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto"
)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
scatter_params = {"s": 10, "alpha": 0.7}

axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=label_codes, cmap="viridis", **scatter_params
)
axes[0].set_title("PCA Visualization", fontsize=16)
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(
    X_tsne[:, 0], X_tsne[:, 1], c=label_codes, cmap="viridis", **scatter_params
)
axes[1].set_title("t-SNE Visualization", fontsize=16)
axes[1].set_xlabel("t-SNE Dimension 1")
axes[1].set_ylabel("t-SNE Dimension 2")

plt.show()
