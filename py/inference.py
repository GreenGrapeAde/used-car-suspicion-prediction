# inference.py
import pickle
import numpy as np
import pandas as pd
import torch
from tabular_dnn import TabularDNN


class SuspectedCarPredictor:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        cat_mapping_path: str,
        config_path: str,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # load category mapping
        with open(cat_mapping_path, "rb") as f:
            self.cat_mapping = pickle.load(f)

        # load config
        with open(config_path, "rb") as f:
            self.config = pickle.load(f)

        # build model
        self.model = TabularDNN(
            cat_cardinalities=self.config["cat_cardinalities"],
            num_dim=len(self.config["num_cols"]),
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"]
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        self.cat_cols = self.config["cat_cols"]
        self.num_cols = self.config["num_cols"]
        self.threshold = self.config["threshold"]

    def _encode_categories(self, df: pd.DataFrame):
        arrs = []
        for c in self.cat_cols:
            categories = self.cat_mapping[c]
            cat_to_idx = {v: i for i, v in enumerate(categories)}
            unk_idx = cat_to_idx.get("__UNK__", len(categories) - 1)

            codes = df[c].astype(str).map(cat_to_idx).fillna(unk_idx)
            arrs.append(codes.astype(int).values)

        return np.stack(arrs, axis=1)

    def preprocess(self, df: pd.DataFrame):
        # numeric
        X_num = self.scaler.transform(df[self.num_cols].astype(float))

        # categorical
        X_cat = self._encode_categories(df)

        x_num = torch.tensor(X_num, dtype=torch.float32).to(self.device)
        x_cat = torch.tensor(X_cat, dtype=torch.long).to(self.device)

        return x_num, x_cat

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, threshold: float = None):
        th = threshold if threshold is not None else self.threshold

        x_num, x_cat = self.preprocess(df)
        logits = self.model((x_num, x_cat))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= th).astype(int)

        result = df.copy()
        result["suspected_prob"] = probs
        result["suspected_pred"] = preds
        return result
