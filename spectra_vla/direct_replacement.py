"""Capacity-matched, base-conditioned direct replacement control."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps)


class BaseConditionedDirectReplacement(nn.Module):
    """A near-capacity-matched base-conditioned direct replacement.

    The target state is generated from [B; s] through an unconstrained
    identity skip.  There is no projection against B, norm-bound, gate,
    fine-branch loss, or q=0 branch.  The skip prevents the rank-32 bottleneck
    from making high base anchoring algebraically impossible.
    """

    def __init__(self, token_dim: int, spec_dim: int, rank: int, n_classes: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.base_norm = nn.LayerNorm(token_dim)
        self.spec_norm = nn.LayerNorm(spec_dim)
        self.base_down = nn.Linear(token_dim, rank)
        self.spec_down = nn.Linear(spec_dim, rank)
        self.up = nn.Linear(rank, token_dim, bias=False)
        self.material_head = nn.Linear(token_dim, n_classes)

    def representation(self, base: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        base_op = layer_norm(base, self.eps)
        hidden = F.gelu(self.base_down(self.base_norm(base_op)) + self.spec_down(self.spec_norm(spec)))
        return layer_norm(base_op + self.up(hidden), self.eps)

    def forward(self, base: torch.Tensor, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.representation(base, spec)
        return self.material_head(z), z


def train_model(
    base_train: np.ndarray,
    spec_train: np.ndarray,
    y_train: np.ndarray,
    base_val: np.ndarray,
    spec_val: np.ndarray,
    y_val: np.ndarray,
    *,
    rank: int,
    alignment: float,
    epochs: int,
    seed: int,
    eps: float,
    sample_weights: np.ndarray,
) -> tuple[BaseConditionedDirectReplacement, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    classes = np.asarray(sorted(set(int(x) for x in y_train.tolist())), dtype=np.int64)
    lookup = {int(label): index for index, label in enumerate(classes.tolist())}
    y_train_i = np.asarray([lookup[int(value)] for value in y_train], dtype=np.int64)
    y_val_i = np.asarray([lookup[int(value)] for value in y_val], dtype=np.int64)
    if not torch.cuda.is_available():
        raise RuntimeError("the retained direct-replacement recipe requires CUDA")
    device = torch.device("cuda")
    model = BaseConditionedDirectReplacement(
        base_train.shape[1], spec_train.shape[1], rank, len(classes), eps
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(y_train_i, minlength=len(classes)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    class_weights_t = torch.from_numpy(class_weights).to(device)
    bt = torch.from_numpy(base_train).float().to(device)
    st = torch.from_numpy(spec_train).float().to(device)
    yt = torch.from_numpy(y_train_i).long().to(device)
    weights_t = torch.from_numpy(sample_weights.astype(np.float32)).to(device)
    bv = torch.from_numpy(base_val).float().to(device)
    sv = torch.from_numpy(spec_val).float().to(device)
    yv = torch.from_numpy(y_val_i).long().to(device)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(bt), device=device)
        total = 0.0
        for start in range(0, len(bt), 256):
            indices = permutation[start:start + 256]
            logits, z = model(bt[indices], st[indices])
            ce = F.cross_entropy(logits, yt[indices], weight=class_weights_t, reduction="none")
            align = (z - layer_norm(bt[indices], eps)).square().mean(dim=1)
            loss = ((ce + float(alignment) * align) * weights_t[indices]).sum()
            loss = loss / weights_t[indices].sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits, z = model(bv, sv)
            val_loss = float(F.cross_entropy(logits, yv, weight=class_weights_t) + float(alignment) * F.mse_loss(z, layer_norm(bv, eps)))
            val_accuracy = float((logits.argmax(1) == yv).float().mean())
        history.append({"epoch": float(epoch + 1), "train_loss": total / len(bt), "validation_loss": val_loss, "validation_class_accuracy": val_accuracy})
    model.eval()
    return model, {"classes": classes.tolist(), "epochs_completed": epochs, "history": history}


@torch.no_grad()
def encode(model: BaseConditionedDirectReplacement, base: np.ndarray, spec: np.ndarray, eps: float) -> np.ndarray:
    device = next(model.parameters()).device
    outputs: list[np.ndarray] = []
    for start in range(0, len(base), 512):
        b = torch.from_numpy(base[start:start + 512]).float().to(device)
        s = torch.from_numpy(spec[start:start + 512]).float().to(device)
        outputs.append(model.representation(b, s).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)
