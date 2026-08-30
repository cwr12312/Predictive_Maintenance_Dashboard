"""
utils/torch_architectures.py
=============================
Exact PyTorch `nn.Module` definitions copied from the original training
notebooks (meta_sgd_model.py, maml.py, fbcl_model.py) so the saved
`state_dict()` checkpoints can be loaded and used for inference without
any retraining. Do not change layer names/shapes here — they must match
the keys stored in the .pth files.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Meta-SGD  (meta_sgd_model.py -> FaultTransformer)
# ---------------------------------------------------------------------
class FaultTransformer(nn.Module):
    """Each input feature becomes a token; self-attention learns feature interactions."""

    def __init__(self, n_feat, n_classes, d_model=64, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_feat, d_model) * 0.01)
        self.drop_in = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x) + self.pos
        x = self.drop_in(x)
        x = self.encoder(x)
        x = self.norm_out(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------
# MAML  (maml.py -> OptimizedmamlTransformer)
# ---------------------------------------------------------------------
class OptimizedmamlTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=128, n_heads=4, n_layers=4, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        self.feature_embedding = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 8,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.BatchNorm1d(d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2), nn.BatchNorm1d(d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self.pretraining_mode = True  # switched to False for inference below

    def forward(self, x, mask=None):
        batch_size, n_feat = x.shape
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            mask_token_expanded = self.mask_token.expand(batch_size, n_feat, self.d_model)
            x = torch.where(mask_expanded, mask_token_expanded, x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.layer_norm(x)

        if self.pretraining_mode:
            reconstructions = self.reconstruction_head(x[:, 1:, :])
            return reconstructions.squeeze(-1)

        cls_representation = x[:, 0]
        return self.classification_head(cls_representation)

    def switch_to_finetune(self):
        self.pretraining_mode = False


# ---------------------------------------------------------------------
# FBCL  (fbcl_model.py -> OptimizedFBCLTransformer)
# ---------------------------------------------------------------------
class OptimizedFBCLTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=256, n_heads=16, n_layers=6, dropout=0.15):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        self.pos_scale = nn.Parameter(torch.ones(1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.BatchNorm1d(d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2), nn.BatchNorm1d(d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self.boosting_heads = nn.ModuleList()

    def add_boosting_head(self):
        head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2), nn.BatchNorm1d(self.d_model * 2), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(self.d_model * 2, self.d_model), nn.BatchNorm1d(self.d_model), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(self.d_model, self.n_classes),
        )
        self.boosting_heads.append(head)
        return head

    def forward(self, x, use_boosting=True):
        batch_size, n_feat = x.shape
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.pos_scale * self.pos_embedding[:, :n_feat, :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)

        logits = self.classifier(x)
        if use_boosting and len(self.boosting_heads) > 0:
            for head in self.boosting_heads:
                logits = logits + head(x)
        return logits


def count_boosting_heads(state_dict: dict) -> int:
    """Infer how many boosting heads were saved in an FBCL checkpoint."""
    indices = set()
    for key in state_dict:
        if key.startswith("boosting_heads."):
            indices.add(int(key.split(".")[1]))
    return len(indices)
