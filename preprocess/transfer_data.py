import logging
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from rich.logging import RichHandler
from rich.progress import track
from scipy import linalg

from features import (
    frequency_domain_features,
    time_domain_features,
    wavelet_packet_features,
)

# 默认设置：48kHz 作为源域训练，12kHz 作为源域验证；目标域用于 DANN（无标签）
SOURCE_TRAIN_DIR = "./数据集/源域数据集/48kHz_DE_data/"
SOURCE_VAL_DIR = "./数据集/源域数据集/12kHz_FE_data/"
TARGET_DATA_DIR = "./数据集/目标域数据集/"
OUTPUT_FILE = "./data/transfer_learning_data.npz"
SAMPLE_LENGTH = 1024

LABEL_MAP = {"B": "Ball", "IR": "Inner Race", "N": "Normal", "OR": "Outer Race"}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
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
# ========== 4.（可选）CORAL 工具 ==========
# ===================================================================


def coral(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + np.eye(d)
    Cs_inv_sqrt = linalg.inv(linalg.sqrtm(Cs))
    A = Cs_inv_sqrt @ linalg.sqrtm(Ct)
    return np.real(Xs @ A)


# ===================================================================
# ========== 5. MAIN EXECUTION SCRIPT ==========
# ===================================================================

if __name__ == "__main__":
    # --- Step 1: 源域训练（48kHz）---
    logger.info("[bold]Step 1: Processing Source-Train (48kHz) ...[/bold]")
    feats_train_df, y_train = process_domain_data(
        SOURCE_TRAIN_DIR, is_source_domain=True
    )
    logger.info(
        f"Source-Train extracted: X={feats_train_df.shape}, y={y_train.shape}, classes={sorted(set(y_train.tolist()))}"
    )

    # --- Step 2: 源域验证（12kHz）---
    logger.info("\n[bold]Step 2: Processing Source-Val (12kHz) ...[/bold]")
    feats_val_df, y_val = process_domain_data(SOURCE_VAL_DIR, is_source_domain=True)
    logger.info(
        f"Source-Val extracted: X={feats_val_df.shape}, y={y_val.shape}, classes={sorted(set(y_val.tolist()))}"
    )

    # --- Step 3: 目标域（无标签，用于 DANN）---
    logger.info("\n[bold]Step 3: Processing Target (Unlabeled) ...[/bold]")
    feats_tgt_df, _ = process_domain_data(TARGET_DATA_DIR, is_source_domain=False)
    logger.info(f"Target extracted: X={feats_tgt_df.shape}")

    # --- 对齐特征列，确保一致顺序 ---
    common_cols = sorted(
        set(feats_train_df.columns)
        & set(feats_val_df.columns)
        & set(feats_tgt_df.columns)
    )
    if not common_cols:
        raise RuntimeError("No common feature columns among train/val/target.")
    feats_train_df = feats_train_df[common_cols]
    feats_val_df = feats_val_df[common_cols]
    feats_tgt_df = feats_tgt_df[common_cols]

    # --- 只保存 RAW（原始）特征，避免任何泄漏；缩放/对齐在训练脚本执行 ---
    X_train_s_raw = feats_train_df.values.astype(np.float64)
    X_val_s_raw = feats_val_df.values.astype(np.float64)
    X_target_raw = feats_tgt_df.values.astype(np.float64)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    logger.info(f"\n[bold]Step 4: Saving arrays to '{OUTPUT_FILE}' ...[/bold]")
    np.savez(
        OUTPUT_FILE,
        X_train_s_raw=X_train_s_raw,
        y_train_s=y_train,
        X_val_s_raw=X_val_s_raw,
        y_val_s=y_val,
        X_target_raw=X_target_raw,
        feature_names=np.array(common_cols),
        source_train_dir=np.array([SOURCE_TRAIN_DIR]),
        source_val_dir=np.array([SOURCE_VAL_DIR]),
        target_dir=np.array([TARGET_DATA_DIR]),
    )
    logger.info("[bold green]✅ Done. Data ready for transfer learning (DANN).")
