import logging
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from rich.logging import RichHandler
from rich.progress import track
from scipy import linalg
from scipy.signal import resample

from features import (
    envelope_spectrum_features,
    frequency_domain_features,
    time_domain_features,
    wavelet_packet_features,
)

# --- 重要参数设置 ---
# 1. 验证集路径修正：确保验证集与训练集同源（都来自驱动端 DE）
# 2. 目标采样率：统一到目标域的 32kHz，是更稳健的策略
# 3. 采样长度：大幅增加到 16384，确保捕捉多次旋转周期
SOURCE_TRAIN_DIR = "./数据集/源域数据集/48kHz_DE_data/"
SOURCE_VAL_DIR = "./数据集/源域数据集/12kHz_DE_data/"  # <-- 【修正】使用DE数据进行验证
TARGET_DATA_DIR = "./数据集/目标域数据集/"
OUTPUT_FILE = "./data/transfer_learning_data.npz"

TARGET_SR = 32000  # <-- 【优化】统一到目标域的采样率
SAMPLE_LENGTH = 16384  # <-- 【关键优化】大幅增加采样长度

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
    parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(parts):
        if part in LABEL_MAP:
            return LABEL_MAP[part]
    return "Unknown"


def process_domain_data(
    root_dir: str,
    original_sr: int,
    target_sr: int,
    is_source_domain: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    all_file_paths = [
        os.path.join(p, f)
        for p, _, files in os.walk(root_dir)
        for f in files
        if f.endswith(".mat")
    ]
    if not all_file_paths:
        raise FileNotFoundError(f"No .mat files were found in: {root_dir}")

    all_features_list = []
    all_labels_list = []

    domain_name = "Source Domain" if is_source_domain else "Target Domain"
    for file_path in track(all_file_paths, description=f"[cyan]Processing {domain_name}"):
        mat_data = sio.loadmat(file_path)
        label = _extract_source_label(file_path) if is_source_domain else "Unknown"

        # --- 【核心升级】寻找数据key的逻辑 ---
        data_keys = []
        if is_source_domain:
            # 【修正】源域：只使用最清晰的驱动端(DE)信号
            data_keys = [k for k in mat_data if "DE_time" in k]
        else:
            # 目标域：名称不规范，寻找最大的那个数组作为主信号
            valid_keys = [
                k
                for k in mat_data
                if isinstance(mat_data[k], np.ndarray) and not k.startswith("__")
            ]
            if valid_keys:
                data_keys = [max(valid_keys, key=lambda k: mat_data[k].size)]

        if not data_keys:
            logger.warning(f"No valid signal found in {os.path.basename(file_path)}. Skipping.")
            continue

        # --- 循环处理该文件中的所有传感器信号 ---
        for key in data_keys:
            signal = mat_data[key].flatten()

            # --- 重采样到目标频率 ---
            if original_sr != target_sr:
                num_samples_new = int(len(signal) * (target_sr / original_sr))
                signal = resample(signal, num_samples_new)

            num_samples = len(signal) // SAMPLE_LENGTH
            if num_samples == 0:
                logger.warning(f"Signal from key '{key}' in {os.path.basename(file_path)} is too short. Skipping.")
                continue

            segments = signal[: num_samples * SAMPLE_LENGTH].reshape(num_samples, SAMPLE_LENGTH)

            # --- 【核心升级】提取所有类型的特征，包括新的包络谱特征 ---
            file_features = {}
            file_features.update(time_domain_features(segments))
            file_features.update(frequency_domain_features(segments, fs=target_sr))
            file_features.update(wavelet_packet_features(segments))
            file_features.update(envelope_spectrum_features(segments, fs=target_sr))

            # 【修正】直接将特征字典转换为DataFrame，不加任何前缀
            df_features = pd.DataFrame(file_features)

            all_features_list.append(df_features)
            all_labels_list.extend([label] * num_samples)

    if not all_features_list:
        raise ValueError(f"No data could be processed in {root_dir}.")

    features_df = pd.concat(all_features_list, ignore_index=True)
    labels_array = np.array(all_labels_list)

    if is_source_domain:
        unique_labels = sorted(list({lab for lab in labels_array if lab != "Unknown"}))
        if len(unique_labels) < 2:
            raise ValueError(f"Source domain has < 2 valid classes: {unique_labels}")

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
    # --- Step 1: 源域训练（原始 48kHz -> 目标 32kHz）---
    logger.info(f"[bold]Step 1: Processing Source-Train (48kHz -> {TARGET_SR}Hz) ...[/bold]")
    feats_train_df, y_train = process_domain_data(
        root_dir=SOURCE_TRAIN_DIR,
        original_sr=48000,
        target_sr=TARGET_SR,
        is_source_domain=True,
    )
    logger.info(
        f"Source-Train extracted: X={feats_train_df.shape}, y={y_train.shape}, classes={sorted(set(y_train.tolist()))}"
    )

    # --- Step 2: 源域验证（原始 12kHz -> 目标 32kHz）---
    logger.info(f"\n[bold]Step 2: Processing Source-Val (12kHz -> {TARGET_SR}Hz) ...[/bold]")
    feats_val_df, y_val = process_domain_data(
        root_dir=SOURCE_VAL_DIR,
        original_sr=12000,
        target_sr=TARGET_SR,
        is_source_domain=True,
    )
    logger.info(
        f"Source-Val extracted: X={feats_val_df.shape}, y={y_val.shape}, classes={sorted(set(y_val.tolist()))}"
    )

    # --- Step 3: 目标域（原始 32kHz -> 目标 32kHz）---
    logger.info(f"\n[bold]Step 3: Processing Target (32kHz -> {TARGET_SR}Hz) ...[/bold]")
    feats_tgt_df, _ = process_domain_data(
        root_dir=TARGET_DATA_DIR,
        original_sr=32000,
        target_sr=TARGET_SR,
        is_source_domain=False,
    )
    logger.info(f"Target extracted: X={feats_tgt_df.shape}")

    # --- 对齐特征列，确保一致顺序 ---
    common_cols = sorted(
        list(set(feats_train_df.columns) & set(feats_val_df.columns))
    )
    # 目标域的特征列可能不完全一样，只保留公共部分
    target_cols = sorted(list(set(common_cols) & set(feats_tgt_df.columns)))
    if not target_cols:
        raise RuntimeError("No common feature columns among train/val/target.")
    
    logger.info(f"Aligning to {len(target_cols)} common feature columns.")
    feats_train_df = feats_train_df[target_cols]
    feats_val_df = feats_val_df[target_cols]
    feats_tgt_df = feats_tgt_df[target_cols]

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
        feature_names=np.array(target_cols),
        source_train_dir=np.array([SOURCE_TRAIN_DIR]),
        source_val_dir=np.array([SOURCE_VAL_DIR]),
        target_dir=np.array([TARGET_DATA_DIR]),
    )
    logger.info("[bold green]✅ Done. Data ready for transfer learning.")