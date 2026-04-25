from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GPULogisticRegression:
    """
    Logistic regression trained entirely on GPU via PyTorch.

    Matches the sklearn predict_proba interface so the rest of the
    pipeline needs no changes.  Scaling is handled externally (same as
    before) so score_all_providers still works unchanged.
    """

    def __init__(self, weight_decay: float = 1e-3, n_epochs: int = 150,
                 batch_size: int = 8192, pos_weight: float = 1.0):
        self.weight_decay = weight_decay
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.pos_weight   = pos_weight
        self._model: nn.Linear | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GPULogisticRegression":
        n_features = X.shape[1]
        self._model = nn.Linear(n_features, 1).to(TORCH_DEVICE)

        pw  = torch.tensor([self.pos_weight], dtype=torch.float32, device=TORCH_DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=1e-2, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs)

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                        pin_memory=True, num_workers=0)

        self._model.train()
        for _ in range(self.n_epochs):
            for Xb, yb in dl:
                Xb, yb = Xb.to(TORCH_DEVICE), yb.to(TORCH_DEVICE)
                optimizer.zero_grad()
                loss = criterion(self._model(Xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            Xt    = torch.tensor(X, dtype=torch.float32).to(TORCH_DEVICE)
            probs = torch.sigmoid(self._model(Xt).squeeze()).cpu().numpy()
        return np.column_stack([1 - probs, probs])
