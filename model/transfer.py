import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from rich.logging import RichHandler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("transfer_learning")


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class DomainFeatureDataset(Dataset):
    def __init__(self, X, y: np.ndarray | None = None):
        """
        Dataset that assumes y is already numeric-encoded (0..C-1).
        Do NOT fit encoders here to avoid train/val mismatch.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], -1  # Return a dummy label


class FeatureTransformerDANN(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        hidden_dim=64,
        lr=1e-4,
        dropout=0.1,
        beta_kl=0.02,
        class_weights: torch.Tensor | None = None,
        warmup_epochs: int = 5,  # <-- 优化点1: 增加warmup参数
    ):
        super().__init__()
        # 保存所有超参数，包括warmup_epochs
        self.save_hyperparameters()
        # 同时存为实例属性，便于访问与静态检查
        self.lr = lr
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

        # 将每个标量特征视为一个token
        self.seq_len = input_dim
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))
        self.norm_input = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器和领域分类器的定义不变
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # 其他初始化部分不变
        if class_weights is not None:
            self.register_buffer("cls_weights", class_weights.float())
        else:
            self.cls_weights = None

        self.criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        # 默认启用 DANN；当且仅当设置 NO_DANN=1 时关闭
        self.use_dann = os.getenv("NO_DANN", "0") != "1"
        self.use_class_weights = os.getenv("USE_CLASS_WEIGHTS", "0") == "1"

    def forward_features(self, x):
        # x: [B, F]
        x = x.unsqueeze(-1)  # [B, F, 1]
        x = self.embedding(x)  # [B, F, D]
        x = x + self.pos_encoder[:, : x.size(1), :]
        x = self.norm_input(x)
        x = self.dropout(x)
        x = self.transformer(x)

        # <-- 优化点2: 使用均值池化替代额外的Attention层
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        source_batch = batch["source"]
        target_batch = batch["target"]
        x_s, y_s = source_batch
        x_t, _ = target_batch

        lambda_grl = 0.0
        if self.current_epoch >= self.warmup_epochs:
            max_epochs = self.trainer.max_epochs or 1
            # DANN生效的总周期数
            effective_max_epochs = max(1, max_epochs - self.warmup_epochs)
            # DANN生效后的当前进度
            current_progress_epoch = self.current_epoch - self.warmup_epochs
            p = float(current_progress_epoch) / float(effective_max_epochs)

            lambda_grl = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1) * 0.5

        f_s = self.forward_features(x_s)
        f_t = self.forward_features(x_t)
        f_concat = torch.cat([f_s, f_t], dim=0)

        logits_s = self.classifier(f_s)
        logits_t = self.classifier(f_t)

        weight = self.cls_weights if self.use_class_weights else None
        loss_label = F.cross_entropy(logits_s, y_s, weight=weight)

        p_s = F.softmax(logits_s, dim=1)
        p_s_mean = p_s.mean(dim=0)
        log_p_t = F.log_softmax(logits_t, dim=1)

        if self.use_dann:
            reversed_features = grad_reverse(f_concat, lambda_grl)
            domain_logits = self.domain_classifier(reversed_features)
            domain_labels = torch.cat(
                [
                    torch.zeros(f_s.size(0), dtype=torch.long, device=self.device),
                    torch.ones(f_t.size(0), dtype=torch.long, device=self.device),
                ]
            )
            loss_domain = F.cross_entropy(domain_logits, domain_labels)
            loss_kl = self.kl_criterion(log_p_t, p_s_mean.detach())
            loss = loss_label + loss_domain + self.beta_kl * loss_kl
        else:
            loss_domain = torch.tensor(0.0, device=self.device)
            loss = loss_label

        self.log_dict(
            {
                "train_loss": loss,
                "loss_label": loss_label,
                "loss_domain": loss_domain,
                "lambda_grl": lambda_grl,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        weight = self.cls_weights if self.use_class_weights else None
        loss = F.cross_entropy(logits, y, weight=weight)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # 使用self.hparams.lr替代self.lr，这是Lightning的推荐做法
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer


logger.info("Loading pre-processed data from 'data/transfer_learning_data.npz' …")
data = np.load("data/transfer_learning_data.npz", allow_pickle=True)

# New schema produced by preprocess/transfer_data.py
X_train_s_raw = data["X_train_s_raw"]
X_val_s_raw = data["X_val_s_raw"]
X_target_raw = data["X_target_raw"]

y_train_s_str = data["y_train_s"]
y_val_s_str = data["y_val_s"]

# 过滤掉无法识别标签的样本，避免把 Unknown 当成类别
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

# Encode labels using training set only, then transform val
le = LabelEncoder().fit(y_train_s_str)
y_train_s = np.asarray(le.transform(y_train_s_str), dtype=np.int64)

# 过滤验证集中不在训练类集合中的样本，避免 transform 报错
known_classes = set(le.classes_.tolist())
mask_val_known = np.array([lbl in known_classes for lbl in y_val_s_str])
if not mask_val_known.all():
    dropped = int((~mask_val_known).sum())
    logger.warning(
        f"Val contains {dropped} samples with unseen labels; dropping them to match train classes {list(known_classes)}"
    )
    y_val_s_str = y_val_s_str[mask_val_known]
    X_val_s_raw = X_val_s_raw[mask_val_known]

y_val_s = np.asarray(le.transform(y_val_s_str), dtype=np.int64)
num_classes = len(le.classes_)
if num_classes < 2:
    raise RuntimeError("Training set has <2 classes. Check dataset directories.")

# Standardize using train-only statistics; target uses its own scaler
scaler_s = StandardScaler().fit(X_train_s_raw)
X_train_s = scaler_s.transform(X_train_s_raw)
X_val_s = scaler_s.transform(X_val_s_raw)

scaler_t = StandardScaler().fit(X_target_raw)
X_target = scaler_t.transform(X_target_raw)

# Guard against potential NaN/Inf values
X_train_s = np.nan_to_num(X_train_s, posinf=1e6, neginf=-1e6)
X_val_s = np.nan_to_num(X_val_s, posinf=1e6, neginf=-1e6)
X_target = np.nan_to_num(X_target, posinf=1e6, neginf=-1e6)


def _coral_fit(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + np.eye(d)
    from scipy import linalg as _lg

    A = _lg.inv(_lg.sqrtm(Cs)) @ _lg.sqrtm(Ct)
    return np.real(A)


# Compute CORAL transform on train split only, then apply to both train and val
A_coral = _coral_fit(X_train_s, X_target)
X_train_s_aligned = np.real(X_train_s @ A_coral)
X_val_s_aligned = np.real(X_val_s @ A_coral)

input_dim = X_train_s_aligned.shape[1]
logger.info(f"Input feature dimension: {input_dim}")
logger.info(f"Classes ({num_classes}): {list(le.classes_)}")
logger.info(
    f"Samples -> source(train)={len(X_train_s_aligned)}, source(val)={len(X_val_s_aligned)}, target={len(X_target)}"
)

train_source_dataset = DomainFeatureDataset(X_train_s_aligned, y_train_s)
val_dataset = DomainFeatureDataset(X_val_s_aligned, y_val_s)
train_target_dataset = DomainFeatureDataset(X_target)  # No labels for target

train_source_loader = DataLoader(train_source_dataset, batch_size=64, shuffle=True)
train_target_loader = DataLoader(train_target_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

train_loaders = CombinedLoader(
    {"source": train_source_loader, "target": train_target_loader},
    mode="max_size_cycle",
)


logger.info("\nInitializing DANN model...")
logger.info(
    f"Batches -> source: {len(train_source_loader)}, target: {len(train_target_loader)}, val: {len(val_loader)}"
)

# Compute inverse-frequency class weights
counts = np.bincount(y_train_s, minlength=num_classes)
freq = counts / counts.sum()
inv_freq = 1.0 / np.maximum(freq, 1e-6)
cls_weights = (inv_freq / inv_freq.sum() * num_classes).astype(np.float32)
logger.info(
    f"Class counts: {counts.tolist()} -> weights: {np.round(cls_weights, 3).tolist()}"
)

model = FeatureTransformerDANN(
    input_dim=input_dim,
    num_classes=num_classes,
    lr=1e-4,
    dropout=0.4,
    class_weights=torch.tensor(cls_weights),
)

pl.seed_everything(42, workers=True)
max_epochs = int(os.getenv("MAX_EPOCHS", "30"))
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices=1,
    callbacks=[RichProgressBar()],
    log_every_n_steps=10,
    gradient_clip_val=1.0,
)

logger.info("Starting model training...")
trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loader)
logger.info("Training complete!")
