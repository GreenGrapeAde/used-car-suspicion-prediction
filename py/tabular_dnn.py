# tabular_dnn.py
import torch
import torch.nn as nn
import numpy as np


class TabularDNN(nn.Module):
    def __init__(
        self,
        cat_cardinalities: dict,
        num_dim: int,
        emb_dim_rule="auto",
        hidden_dims=(256, 128),
        dropout=0.3,
        n_classes=2
    ):
        super().__init__()

        self.cat_cols = list(cat_cardinalities.keys())
        self.num_dim = num_dim

        # Embedding layers
        self.embs = nn.ModuleList()
        emb_out_dim_total = 0

        for c in self.cat_cols:
            n_cat = cat_cardinalities[c]
            if emb_dim_rule == "auto":
                emb_dim = int(min(50, round(np.sqrt(n_cat)) + 1))
            else:
                emb_dim = int(emb_dim_rule)

            self.embs.append(nn.Embedding(n_cat, emb_dim))
            emb_out_dim_total += emb_dim

        in_dim = num_dim + emb_out_dim_total

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x_num, x_cat = x  # x_cat: (B, n_cat_cols)

        emb_list = []
        for i, emb in enumerate(self.embs):
            emb_list.append(emb(x_cat[:, i]))

        if len(emb_list) > 0:
            x_emb = torch.cat(emb_list, dim=1)
        else:
            x_emb = torch.empty((x_num.size(0), 0), device=x_num.device)

        x_all = torch.cat([x_num, x_emb], dim=1)
        logits = self.mlp(x_all)
        return logits
