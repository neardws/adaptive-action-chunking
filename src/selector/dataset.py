"""
Dataset for k-Selector training.
Loads oracle-labeled episodes: (siglip_features, state, oracle_k).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .model import K_TO_IDX


class OracleLabelDataset(Dataset):
    """
    Each item: (features [1152], state [D], label [int])
    Labels come from generate_oracle_labels.py output.
    """

    def __init__(
        self,
        features_dir: str,     # dir with .npz files (features + state per episode)
        labels_path: str,      # jsonl from generate_oracle_labels.py
        feature_dim: int = 1152,
        state_dim: int = 32,
    ):
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.items = []

        labels_by_ep = {}
        with open(labels_path) as f:
            for line in f:
                d = json.loads(line)
                labels_by_ep[d["episode"]] = d["k_labels"]

        features_dir = Path(features_dir)
        for ep_file in sorted(features_dir.glob("*.npz")):
            ep_name = ep_file.stem
            if ep_name not in labels_by_ep:
                continue
            data = np.load(ep_file)
            features = data["features"]   # [T, feature_dim]
            states = data["states"]       # [T, state_dim]
            k_labels = labels_by_ep[ep_name]
            T = min(len(features), len(k_labels))
            for t in range(T):
                self.items.append((
                    features[t].astype(np.float32),
                    states[t].astype(np.float32),
                    K_TO_IDX[k_labels[t]],
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, state, label = self.items[idx]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(state),
            torch.tensor(label, dtype=torch.long),
        )
