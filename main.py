import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# ================= Logger =================
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("pl_transformer")


# ================= Dataset =================
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= Lightning Module =================
class FeatureTransformerPL(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        lr=1e-3,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Embedding ---
        self.embedding = nn.Linear(1, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_input = nn.LayerNorm(embed_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Attention Pooling ---
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(embed_dim)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embedding(x)
        x = self.norm_input(x)
        x = self.dropout(x)

        x = self.transformer(x)  # [batch, seq_len, embed_dim]

        # Attention pooling
        attn_output, attn_weights = self.attention(x, x, x)  # Self-attention
        x = self.norm_attn(attn_output)
        x = x.mean(dim=1)  # 平均池化 token

        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


# ================= Load PCA Features =================
data = np.load("./data/pca_features_labels.npz")
X = data["X_pca"]
labels = data["labels"]
logger.info(f"PCA features loaded: X={X.shape}, labels={labels.shape}")

y = LabelEncoder().fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = FeatureDataset(X_train, y_train)
val_dataset = FeatureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ================= Trainer =================
model = FeatureTransformerPL(num_classes=len(np.unique(y)))
trainer = pl.Trainer(max_epochs=20, callbacks=[RichProgressBar()])
trainer.fit(model, train_loader, val_loader)
