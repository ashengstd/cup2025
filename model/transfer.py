import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Use LabelEncoder to convert string labels to integers
        if y is not None:
            self.le = LabelEncoder().fit(y)
            self.y = torch.tensor(self.le.transform(y), dtype=torch.long)
            self.classes = self.le.classes_
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        # For target domain, we only return features
        else:
            return self.X[idx], -1  # Return a dummy label


# ===================================================================
# ========== 3. DANN MODEL (No changes needed) ======================
# ===================================================================
class FeatureTransformerDANN(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        lr=1e-3,
        dropout=0.1,
        beta_kl=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor
        # <<< MODIFIED >>>: Changed nn.Linear to accept multi-dimensional features
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, 100, embed_dim)
        )  # Positional encoding
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
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(embed_dim)

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

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.beta_kl = beta_kl

    def forward_features(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = x + self.pos_encoder[:, : x.size(1), :]
        x = self.norm_input(x)
        x = self.dropout(x)
        x = self.transformer(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm_attn(attn_output)
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

        # Dynamically adjust lambda for GRL
        p = float(self.current_epoch) / float(self.trainer.max_epochs)
        lambda_grl = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1

        f_s = self.forward_features(x_s)
        f_t = self.forward_features(x_t)
        f_concat = torch.cat([f_s, f_t], dim=0)

        logits_s = self.classifier(f_s)
        logits_t = self.classifier(f_t)
        loss_label = self.criterion(logits_s, y_s)
        p_s = F.softmax(logits_s, dim=1)
        p_s_mean = p_s.mean(dim=0)
        log_p_t = F.log_softmax(logits_t, dim=1)
        reversed_features = grad_reverse(f_concat, lambda_grl)
        domain_logits = self.domain_classifier(reversed_features)
        domain_labels = torch.cat(
            [
                torch.zeros(f_s.size(0), dtype=torch.long, device=self.device),
                torch.ones(f_t.size(0), dtype=torch.long, device=self.device),
            ]
        )
        loss_domain = self.criterion(domain_logits, domain_labels)
        loss_kl = self.kl_criterion(log_p_t, p_s_mean.detach())
        loss = loss_label + loss_domain + self.beta_kl * loss_kl
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
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


logger.info("Loading pre-processed data from 'transfer_learning_data.npz'...")
data = np.load("./data/transfer_learning_data.npz")

X_source_aligned = data["X_source_aligned"]
y_source = data["y_source"]
X_target = data["X_target"]

num_classes = len(np.unique(y_source))
input_dim = X_source_aligned.shape[1]

logger.info(f"Input feature dimension: {input_dim}")
logger.info(f"Number of classes: {num_classes}")
logger.info(f"Source samples: {len(X_source_aligned)}, Target samples: {len(X_target)}")

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_source_aligned, y_source, test_size=0.2, random_state=42, stratify=y_source
)

train_source_dataset = DomainFeatureDataset(X_train_s, y_train_s)
val_dataset = DomainFeatureDataset(X_val_s, y_val_s)
train_target_dataset = DomainFeatureDataset(X_target)  # No labels for target

train_source_loader = DataLoader(train_source_dataset, batch_size=64, shuffle=True)
train_target_loader = DataLoader(train_target_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

train_loaders = {"source": train_source_loader, "target": train_target_loader}


logger.info("\nInitializing DANN model...")
model = FeatureTransformerDANN(
    input_dim=input_dim, num_classes=num_classes, lr=1e-4, dropout=0.4
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1,
    callbacks=[RichProgressBar()],
    log_every_n_steps=10,
)

logger.info("Starting model training...")
trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loader)
logger.info("Training complete!")
