import logging
import os

import numpy as np
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("svm_transfer")


def _coral_fit_transform(Xs: np.ndarray, Xt: np.ndarray):
    """Fit CORAL transform on Xs using Xt as target domain; return transformed Xs and the transform A."""
    from scipy import linalg as _lg

    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + np.eye(d)
    A = _lg.inv(_lg.sqrtm(Cs)) @ _lg.sqrtm(Ct)
    Xs_aligned = np.real(Xs @ A)
    return Xs_aligned, A


def main():
    logger.info("Loading pre-processed data from 'data/transfer_learning_data.npz' …")
    data = np.load("data/transfer_learning_data.npz", allow_pickle=True)

    X_train_s_raw = data["X_train_s_raw"]
    X_val_s_raw = data["X_val_s_raw"]
    X_target_raw = data["X_target_raw"]

    y_train_s_str = data["y_train_s"]
    y_val_s_str = data["y_val_s"]

    # 1) 过滤 Unknown
    if (y_train_s_str == "Unknown").any():
        mask = y_train_s_str != "Unknown"
        X_train_s_raw = X_train_s_raw[mask]
        y_train_s_str = y_train_s_str[mask]
        logger.warning(f"Filtered Unknown from train: remaining={len(y_train_s_str)}")
    if (y_val_s_str == "Unknown").any():
        mask = y_val_s_str != "Unknown"
        X_val_s_raw = X_val_s_raw[mask]
        y_val_s_str = y_val_s_str[mask]
        logger.warning(f"Filtered Unknown from val: remaining={len(y_val_s_str)}")

    # 2) LabelEncoder 拟合训练集类
    le = LabelEncoder().fit(y_train_s_str)
    y_train_s = np.asarray(le.transform(y_train_s_str), dtype=np.int64)

    known = set(le.classes_.tolist())
    mask_val_known = np.array([lbl in known for lbl in y_val_s_str])
    if not mask_val_known.all():
        dropped = int((~mask_val_known).sum())
        logger.warning(
            f"Val contains {dropped} unseen-label samples; dropping them. Train classes={list(known)}"
        )
        y_val_s_str = y_val_s_str[mask_val_known]
        X_val_s_raw = X_val_s_raw[mask_val_known]

    y_val_s = np.asarray(le.transform(y_val_s_str), dtype=np.int64)

    # 3) 标准化：source 用 train 拟合，target 用自身拟合
    scaler_s = StandardScaler().fit(X_train_s_raw)
    X_train_s = scaler_s.transform(X_train_s_raw)
    X_val_s = scaler_s.transform(X_val_s_raw)

    scaler_t = StandardScaler().fit(X_target_raw)
    X_target = scaler_t.transform(X_target_raw)

    X_train_s = np.nan_to_num(X_train_s, posinf=1e6, neginf=-1e6)
    X_val_s = np.nan_to_num(X_val_s, posinf=1e6, neginf=-1e6)
    X_target = np.nan_to_num(X_target, posinf=1e6, neginf=-1e6)

    # 4) CORAL 对齐（在 train 上拟合，应用到 train 与 val）
    X_train_s_aligned, A = _coral_fit_transform(X_train_s, X_target)
    X_val_s_aligned = np.real(X_val_s @ A)

    logger.info(
        f"Shapes -> train_s: {X_train_s_aligned.shape}, val_s: {X_val_s_aligned.shape}, target: {X_target.shape}"
    )
    logger.info(f"Classes: {list(le.classes_)}")

    # 5) 可选：下采样以加速 SVM 训练（通过环境变量控制）
    max_train = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
    if max_train > 0 and X_train_s_aligned.shape[0] > max_train:
        X_train_s_aligned = X_train_s_aligned[:max_train]
        y_train_s = y_train_s[:max_train]
        logger.warning(f"Subsample train to {max_train} samples for speed.")

    # 6) 训练 SVM（默认 linear，可通过 SVM_KERNEL=rbf 切换）
    kernel = os.getenv("SVM_KERNEL", "linear").lower()
    use_class_weights = os.getenv("USE_CLASS_WEIGHTS", "0") == "1"
    class_weight = "balanced" if use_class_weights else None

    if kernel == "linear":
        clf = LinearSVC(C=1.0, class_weight=class_weight)
    else:
        # rbf 时可配合 gamma 与 C 调参，默认即可
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=class_weight)

    logger.info(f"Training SVM (kernel={kernel}, class_weight={class_weight}) …")
    clf.fit(X_train_s_aligned, y_train_s)

    # 7) 在源域验证集上评估
    y_pred = clf.predict(X_val_s_aligned)
    acc = accuracy_score(y_val_s, y_pred)
    logger.info(f"Val Accuracy: {acc:.4f}")
    report = classification_report(y_val_s, y_pred, target_names=le.classes_)
    cm_text = np.array2string(confusion_matrix(y_val_s, y_pred))
    logger.info("Classification Report:\n%s", report)
    logger.info("Confusion Matrix:\n%s", cm_text)


if __name__ == "__main__":
    main()
