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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("mlp_transfer")


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


class FeatureMLPDANN(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=64,
        lr=1e-4,
        dropout=0.2,
        beta_kl=0.02,
        class_weights: torch.Tensor | None = None,
        warmup_epochs: int = 5,
        lambda_scale: float = 1.0,  # <-- 新增：对抗强度调节器
    ):
        super().__init__()
        self.save_hyperparameters()  # lambda_scale 会被自动保存
        self.lr = lr
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

        # --- 新的MLP特征提取器 ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- 分类器和领域分类器 ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        if class_weights is not None:
            self.register_buffer("cls_weights", class_weights.float())
        else:
            self.cls_weights = None

        self.criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.use_dann = os.getenv("NO_DANN", "0") != "1"
        self.use_class_weights = os.getenv("USE_CLASS_WEIGHTS", "0") == "1"

    def forward_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            source_batch = batch["source"]
            target_batch = batch["target"]
            x_s, y_s = source_batch
            x_t, _ = target_batch

            lambda_grl = 0.0
            if self.current_epoch >= self.warmup_epochs:
                max_epochs = self.trainer.max_epochs or 1
                effective_max_epochs = max(1, max_epochs - self.warmup_epochs)
                current_progress_epoch = self.current_epoch - self.warmup_epochs
                p = float(current_progress_epoch) / float(effective_max_epochs)
                # 【核心修改】使用 lambda_scale 调节对抗强度
                lambda_grl = (
                    (2.0 / (1.0 + np.exp(-10.0 * p)) - 1)
                    * 0.5
                    * self.hparams.lambda_scale
                )

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
        else:
            x_s, y_s = batch
            logits_s = self(x_s)
            weight = self.cls_weights if self.use_class_weights else None
            loss_label = F.cross_entropy(logits_s, y_s, weight=weight)
            loss = loss_label
            loss_domain = torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                "train_loss": loss,
                "loss_label": loss_label,
                "loss_domain": loss_domain,
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer


logger.info("Loading pre-processed data from 'data/transfer_learning_data.npz' …")
data = np.load("data/transfer_learning_data.npz", allow_pickle=True)

X_train_s_raw = data["X_train_s_raw"]
X_val_s_raw = data["X_val_s_raw"]
X_target_raw = data["X_target_raw"]

y_train_s_str = data["y_train_s"]
y_val_s_str = data["y_val_s"]

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

le = LabelEncoder().fit(y_train_s_str)
y_train_s = np.asarray(le.transform(y_train_s_str), dtype=np.int64)

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

scaler_s = StandardScaler().fit(X_train_s_raw)
X_train_s = scaler_s.transform(X_train_s_raw)
X_val_s = scaler_s.transform(X_val_s_raw)

scaler_t = StandardScaler().fit(X_target_raw)
X_target = scaler_t.transform(X_target_raw)

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


if os.getenv("NO_CORAL", "0") != "1":
    logger.info("Applying CORAL alignment...")
    A_coral = _coral_fit(X_train_s, X_target)
    X_train_s_aligned = np.real(X_train_s @ A_coral)
    X_val_s_aligned = np.real(X_val_s @ A_coral)
else:
    logger.warning("Skipping CORAL alignment as per NO_CORAL=1.")
    X_train_s_aligned = X_train_s
    X_val_s_aligned = X_val_s

input_dim = X_train_s_aligned.shape[1]
logger.info(f"Input feature dimension: {input_dim}")
logger.info(f"Classes ({num_classes}): {list(le.classes_)}")
logger.info(
    f"Samples -> source(train)={len(X_train_s_aligned)}, source(val)={len(X_val_s_aligned)}, target={len(X_target)}"
)

train_source_dataset = DomainFeatureDataset(X_train_s_aligned, y_train_s)
val_dataset = DomainFeatureDataset(X_val_s_aligned, y_val_s)
train_target_dataset = DomainFeatureDataset(X_target)

train_source_loader = DataLoader(train_source_dataset, batch_size=64, shuffle=True)
train_target_loader = DataLoader(train_target_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

logger.info("\nInitializing DANN model...")

counts = np.bincount(y_train_s, minlength=num_classes)
freq = counts / counts.sum()
inv_freq = 1.0 / np.maximum(freq, 1e-6)
cls_weights = (inv_freq / inv_freq.sum() * num_classes).astype(np.float32)
logger.info(
    f"Class counts: {counts.tolist()} -> weights: {np.round(cls_weights, 3).tolist()}"
)

model = FeatureMLPDANN(
    input_dim=input_dim,
    num_classes=num_classes,
    lr=1e-4,
    dropout=0.2,
    class_weights=torch.tensor(cls_weights),
    lambda_scale=0.2,
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

if os.getenv("SOURCE_ONLY", "0") == "1":
    logger.warning("<<<<< SOURCE-ONLY SANITY CHECK MODE >>>>>")
    logger.warning("Training on source data only. CombinedLoader is disabled.")
    trainer.fit(
        model, train_dataloaders=train_source_loader, val_dataloaders=val_loader
    )
else:
    logger.info("Starting model training with CombinedLoader (Source + Target)...")
    train_loaders = CombinedLoader(
        {"source": train_source_loader, "target": train_target_loader},
        mode="max_size_cycle",
    )
    trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loader)

logger.info("Training complete!")

logger.info("\n" + "=" * 20 + " FINAL VALIDATION " + "=" * 20)

logger.info("Running prediction on validation set...")
prediction_batches = trainer.predict(model, dataloaders=val_loader)

y_pred_logits = torch.cat(prediction_batches)
y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

y_true = y_val_s

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
cm = confusion_matrix(y_true, y_pred)

logger.info(f"Final Validation Accuracy: {acc:.4f}")
logger.info(f"Final Classification Report:\n{report}")
logger.info(f"Final Confusion Matrix:\n{cm}")
