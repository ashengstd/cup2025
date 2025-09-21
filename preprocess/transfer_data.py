import logging
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from rich.logging import RichHandler
from rich.progress import track
from scipy import linalg
from sklearn.preprocessing import StandardScaler

from features import (
    frequency_domain_features,
    time_domain_features,
    wavelet_packet_features,
)

SOURCE_DATA_DIR = "./数据集/源域数据集/48kHz_DE_data/"
TARGET_DATA_DIR = "./数据集/目标域数据集/"
OUTPUT_FILE = "./data/transfer_learning_data.npz"
SAMPLE_LENGTH = 1024

LABEL_MAP = {"B": "Ball", "IR": "Inner Race", "N": "Normal", "OR": "Outer Race"}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("DataPrep")


def _extract_source_label(file_path: str) -> str:
    """
    Robustly extract the fault type label key (B/IR/OR/N) from a full file path.
    It searches the path components from right to left and returns the first
    directory name that matches a known label key.
    """
    parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(parts):
        if part in LABEL_MAP:
            return LABEL_MAP[part]
    return "Unknown"


def process_domain_data(
    root_dir: str, is_source_domain: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Processes all .mat files in a directory to extract features and labels.
    This function is designed to handle both source and target domain structures.
    """
    all_file_paths = [
        os.path.join(p, f)
        for p, _, files in os.walk(root_dir)
        for f in files
        if f.endswith(".mat")
    ]
    if not all_file_paths:
        raise FileNotFoundError(
            f"No .mat files were found in the directory: {root_dir}"
        )

    all_features_list = []
    all_labels_list = []

    domain_name = "Source Domain" if is_source_domain else "Target Domain"
    for file_path in track(
        all_file_paths, description=f"[cyan]Processing {domain_name}"
    ):
        mat_data = sio.loadmat(file_path)

        # --- Smartly find the data key ---
        data_key = None
        if is_source_domain:
            # For CWRU source data, the drive-end signal is standard.
            data_key = next((k for k in mat_data if "DE" in k), None)
        else:
            # For target data, find the largest array, as names are inconsistent.
            valid_keys = [
                k
                for k in mat_data
                if isinstance(mat_data[k], np.ndarray) and not k.startswith("__")
            ]
            if valid_keys:
                data_key = max(valid_keys, key=lambda k: mat_data[k].size)

        if not data_key:
            logger.warning(
                f"No valid signal found in {os.path.basename(file_path)}. Skipping."
            )
            continue

        # --- Get label ---
        label = "Unknown"
        if is_source_domain:
            # For source, extract label from any path component (e.g., B/IR/OR/N).
            label = _extract_source_label(file_path)

        signal = mat_data[data_key].flatten()
        num_samples = len(signal) // SAMPLE_LENGTH
        if num_samples == 0:
            logger.warning(
                f"Signal in {os.path.basename(file_path)} is too short. Skipping."
            )
            continue

        # --- Segment signal and extract features ---
        segments = signal[: num_samples * SAMPLE_LENGTH].reshape(
            num_samples, SAMPLE_LENGTH
        )

        file_features = {}
        file_features.update(time_domain_features(segments))
        file_features.update(frequency_domain_features(segments))
        file_features.update(wavelet_packet_features(segments))

        all_features_list.append(pd.DataFrame(file_features))
        all_labels_list.extend([label] * num_samples)

    if not all_features_list:
        raise ValueError(f"No data could be processed in {root_dir}.")

    features_df = pd.concat(all_features_list, ignore_index=True)
    labels_array = np.array(all_labels_list)

    # Safety check: ensure multi-class for source domain
    if is_source_domain:
        unique_labels = sorted(list({lab for lab in labels_array if lab != "Unknown"}))
        if len(unique_labels) < 2:
            raise ValueError(
                "Source domain appears to have < 2 valid classes. "
                "Please verify the dataset root and directory structure. "
                f"Detected labels (excluding 'Unknown'): {unique_labels}"
            )
    return features_df, labels_array


# ===================================================================
# ========== 4. CORAL ALGORITHM ==========
# ===================================================================


def coral(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """
    Implements the CORAL algorithm for Unsupervised Domain Adaptation.
    It aligns the covariance of the source data with the covariance of the target data.

    Args:
        Xs: Source domain feature matrix (n_source, n_features).
        Xt: Target domain feature matrix (n_target, n_features).

    Returns:
        The transformed source domain features (Xs_aligned).
    """
    d = Xs.shape[1]
    # Source covariance
    Cs = np.cov(Xs, rowvar=False) + np.eye(d)
    # Target covariance
    Ct = np.cov(Xt, rowvar=False) + np.eye(d)

    # Calculate the transformation matrix
    Cs_inv_sqrt = linalg.inv(linalg.sqrtm(Cs))
    A = Cs_inv_sqrt @ linalg.sqrtm(Ct)

    # Apply the transformation and return the real part
    Xs_aligned = np.real(Xs @ A)
    return Xs_aligned


# ===================================================================
# ========== 5. MAIN EXECUTION SCRIPT ==========
# ===================================================================

if __name__ == "__main__":
    # --- Step 1: Process Labeled Source Domain ---
    logger.info("[bold]Step 1: Processing Source Domain Data...[/bold]")
    features_df_source, labels_source = process_domain_data(
        SOURCE_DATA_DIR, is_source_domain=True
    )
    # Keep a raw copy to avoid future data leakage during training split
    X_source_raw = features_df_source.values.astype(np.float64)
    scaler_source = StandardScaler()
    X_source = scaler_source.fit_transform(X_source_raw)
    logger.info(f"Source data processed. Shape: {X_source.shape}")

    # --- Step 2: Process Unlabeled Target Domain ---
    logger.info("\n[bold]Step 2: Processing Target Domain Data...[/bold]")
    features_df_target, _ = process_domain_data(TARGET_DATA_DIR, is_source_domain=False)
    # Keep a raw copy to avoid future data leakage during training split
    X_target_raw = features_df_target.values.astype(np.float64)
    scaler_target = StandardScaler()
    X_target = scaler_target.fit_transform(X_target_raw)
    logger.info(f"Target data processed. Shape: {X_target.shape}")

    # --- Step 3: Apply CORAL for Domain Alignment ---
    logger.info("\n[bold]Step 3: Applying CORAL for Domain Alignment...[/bold]")
    X_source_aligned = coral(X_source, X_target)
    logger.info(f"CORAL alignment complete. Shape: {X_source_aligned.shape}")

    # --- Step 4: Save Processed Data for DNN Training ---
    logger.info(f"\n[bold]Step 4: Saving all arrays to '{OUTPUT_FILE}'...[/bold]")
    np.savez(
        OUTPUT_FILE,
        # Scaled versions (legacy)
        X_source=X_source,
        X_target=X_target,
        X_source_aligned=X_source_aligned,
        # Raw versions (preferred for training to avoid leakage)
        X_source_raw=X_source_raw,
        X_target_raw=X_target_raw,
        # Labels
        y_source=labels_source,
    )
    logger.info(
        "[bold green]✅ Success![/bold green] Data saved and ready for DNN training."
    )
