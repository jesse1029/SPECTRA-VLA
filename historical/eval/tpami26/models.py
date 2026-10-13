"""Parameter-counted token baselines used by the matched TPAMI ablation."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.tpami26.protocol import OPERATIONAL_LN_EPS


def torch_layer_norm(x: torch.Tensor, eps: float = OPERATIONAL_LN_EPS) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps)


class LowRankTokenMap(nn.Module):
    def __init__(self, input_dim: int, token_dim: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, token_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.down(self.norm(x)))
        return self.up(hidden), hidden


class MatchedTokenMethod(nn.Module):
    """One implementation surface for all learned matched baselines.

    The auxiliary heads are used only to train the representation.  Final table
    predictions always come from the same external readout/prototype code.
    """

    def __init__(
        self, method: str, spec_dim: int, token_dim: int, rank: int,
        n_classes: int, fine_budget: float | None = None,
    ):
        super().__init__()
        self.method = method
        self.spec_dim = spec_dim
        self.token_dim = token_dim
        self.rank = rank
        self.n_classes = n_classes
        self.fine_budget = fine_budget
        in_dim = token_dim + spec_dim if method == "concatenation_projection" else spec_dim
        self.mapper = LowRankTokenMap(in_dim, token_dim, rank)
        self.material_head = nn.Linear(token_dim, n_classes)
        self.fine_head = nn.Linear(token_dim, n_classes)
        self.gate = nn.Linear(rank, 1) if method == "gated_additive_residual" else None

    def representation(
        self, base: torch.Tensor, spec: torch.Tensor, dosage: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_op = torch_layer_norm(base)
        mapper_input = torch.cat([base_op, spec], dim=-1) if self.method == "concatenation_projection" else spec
        fine, hidden = self.mapper(mapper_input)

        if self.method == "concatenation_projection":
            z = torch_layer_norm(fine)
        elif self.method in {
            "direct_hsi_token_projection",
            "spectral_fine_only",
            "parameter_matched_spectral_mlp_projection",
        }:
            z = torch_layer_norm(fine)
        elif self.method == "direct_hsi_to_rgb_token_retargeting":
            z = torch_layer_norm(fine)
        else:
            if self.method in {
                "proposed_base_fine", "proposed_without_fine_loss",
                "proposed_without_layernorm",
            }:
                fine = fine - (fine * base_op).sum(-1, keepdim=True) * base_op / (
                    base_op.square().sum(-1, keepdim=True) + 1e-6
                )
                if self.fine_budget is not None:
                    target_norm = float(self.fine_budget) * base_op.norm(dim=-1, keepdim=True)
                    fine = fine * target_norm / (fine.norm(dim=-1, keepdim=True) + 1e-6)
            if self.method == "gated_additive_residual":
                assert self.gate is not None
                fine = torch.sigmoid(self.gate(hidden)) * fine
            is_zero = isinstance(dosage, (float, int)) and float(dosage) == 0.0
            if is_zero:
                z = base_op
            elif self.method == "proposed_without_layernorm":
                z = base_op + dosage * fine
            else:
                z = torch_layer_norm(base_op + dosage * fine)
        return z, fine, base_op

    def forward(
        self, base: torch.Tensor, spec: torch.Tensor, dosage: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, fine, base_op = self.representation(base, spec, dosage)
        return self.material_head(z), self.fine_head(fine), z, fine, base_op


class LabelEmbeddingResidual(nn.Module):
    def __init__(self, token_dim: int, rank: int, n_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, rank)
        self.up = nn.Linear(rank, token_dim, bias=False)
        self.head = nn.Linear(token_dim, n_classes)

    def representation(
        self, base: torch.Tensor, labels: torch.Tensor, dosage: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_op = torch_layer_norm(base)
        fine = self.up(self.embedding(labels))
        fine = fine - (fine * base_op).sum(-1, keepdim=True) * base_op / (
            base_op.square().sum(-1, keepdim=True) + 1e-6
        )
        z = base_op if float(dosage) == 0.0 else torch_layer_norm(base_op + float(dosage) * fine)
        return z, fine, base_op

    def forward(self, base: torch.Tensor, labels: torch.Tensor, dosage: float):
        z, fine, base_op = self.representation(base, labels, dosage)
        return self.head(z), z, fine, base_op


@dataclass
class TrainSettings:
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    fine_loss_weight: float = 1.0
    base_loss_weight: float = 0.1
    fine_l2_weight: float = 1e-4
    dosage_max: float = 4.0
    patience: int = 10
    restore_best: bool = False
    fine_budget: float | None = None
    direct_align_weight: float = 1.0
    device: str = "cuda"


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _class_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(set(int(x) for x in y.tolist())), dtype=np.int64)
    lookup = {int(c): i for i, c in enumerate(classes.tolist())}
    return classes, np.asarray([lookup[int(v)] for v in y], dtype=np.int64)


def train_token_method(
    method: str,
    base_train: np.ndarray,
    spec_train: np.ndarray,
    y_train: np.ndarray,
    base_val: np.ndarray,
    spec_val: np.ndarray,
    y_val: np.ndarray,
    rank: int,
    seed: int,
    settings: TrainSettings,
    randomize_labels: bool = False,
    sample_weights: np.ndarray | None = None,
) -> tuple[MatchedTokenMethod, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    classes, y_train_i = _class_targets(y_train)
    lookup = {int(c): i for i, c in enumerate(classes.tolist())}
    y_val_i = np.asarray([lookup[int(v)] for v in y_val], dtype=np.int64)
    supervision = y_train_i.copy()
    validation_targets = y_val_i.copy()
    if randomize_labels:
        supervision = np.random.default_rng(seed + 991).permutation(supervision)
        validation_targets = np.random.default_rng(seed + 1991).permutation(validation_targets)

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = MatchedTokenMethod(
        method, spec_train.shape[1], base_train.shape[1], rank, len(classes),
        fine_budget=settings.fine_budget,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    counts = np.bincount(supervision, minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    weights_t = torch.from_numpy(weights).to(device)

    bt = torch.from_numpy(base_train).float().to(device)
    st = torch.from_numpy(spec_train).float().to(device)
    yt = torch.from_numpy(supervision).long().to(device)
    if sample_weights is None:
        sample_weights = np.ones(len(y_train), dtype=np.float32)
    if len(sample_weights) != len(y_train):
        raise ValueError("sample_weights must align with y_train")
    sample_weights_t = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)).to(device)
    bv = torch.from_numpy(base_val).float().to(device)
    sv = torch.from_numpy(spec_val).float().to(device)
    yv = torch.from_numpy(validation_targets).long().to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        model.train()
        perm = torch.randperm(len(bt), device=device)
        total = 0.0
        for start in range(0, len(bt), settings.batch_size):
            idx = perm[start:start + settings.batch_size]
            dosage = torch.rand((len(idx), 1), device=device) * settings.dosage_max
            logits, fine_logits, z, fine, base_op = model(bt[idx], st[idx], dosage)
            if method == "direct_hsi_to_rgb_token_retargeting":
                task_per_sample = F.cross_entropy(
                    logits, yt[idx], weight=weights_t, reduction="none"
                )
                align_per_sample = (z - base_op).square().mean(dim=1)
                per_sample = task_per_sample + float(settings.direct_align_weight) * align_per_sample
                loss = (per_sample * sample_weights_t[idx]).sum() / sample_weights_t[idx].sum().clamp_min(1e-6)
            else:
                per_sample = F.cross_entropy(logits, yt[idx], weight=weights_t, reduction="none")
                loss = (per_sample * sample_weights_t[idx]).sum() / sample_weights_t[idx].sum().clamp_min(1e-6)
                if method != "proposed_without_fine_loss":
                    fine_per_sample = F.cross_entropy(fine_logits, yt[idx], weight=weights_t, reduction="none")
                    loss = loss + settings.fine_loss_weight * (
                        fine_per_sample * sample_weights_t[idx]
                    ).sum() / sample_weights_t[idx].sum().clamp_min(1e-6)
                logits0, _, _, _, _ = model(bt[idx], st[idx], 0.0)
                base_per_sample = F.cross_entropy(logits0, yt[idx], weight=weights_t, reduction="none")
                loss = loss + settings.base_loss_weight * (
                    base_per_sample * sample_weights_t[idx]
                ).sum() / sample_weights_t[idx].sum().clamp_min(1e-6)
                loss = loss + settings.fine_l2_weight * fine.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            logits, fine_logits, z, fine, base_op = model(bv, sv, 1.0)
            if method == "direct_hsi_to_rgb_token_retargeting":
                val_loss = float(
                    F.cross_entropy(logits, yv, weight=weights_t)
                    + float(settings.direct_align_weight) * F.mse_loss(z, base_op)
                )
                val_acc = float((logits.argmax(1) == yv).float().mean())
            else:
                val_loss = float(F.cross_entropy(logits, yv, weight=weights_t))
                val_acc = float((logits.argmax(1) == yv).float().mean())
        history.append({"epoch": float(epoch), "train_loss": total / len(bt), "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no finite model selected for {method}")
    if settings.restore_best:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "classes": classes.tolist(), "best_val_loss": best_loss,
        "epochs_completed": len(history), "history": history,
        "checkpoint_selection": "best_validation" if settings.restore_best else "fixed_final_epoch",
        "trainable_parameters": parameter_count(model), "randomize_labels": bool(randomize_labels),
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
        "sample_weight_mean": float(np.mean(sample_weights)),
        "training_supervision_sha256": hashlib.sha256(supervision.tobytes()).hexdigest(),
        "validation_supervision_sha256": hashlib.sha256(validation_targets.tobytes()).hexdigest(),
        "randomized_validation_labels": bool(randomize_labels),
        "fine_budget": settings.fine_budget,
        "direct_align_weight": settings.direct_align_weight,
    }


def train_label_embedding(
    base_train: np.ndarray,
    y_train: np.ndarray,
    base_val: np.ndarray,
    y_val: np.ndarray,
    rank: int,
    seed: int,
    settings: TrainSettings,
    sample_weights: np.ndarray | None = None,
) -> tuple[LabelEmbeddingResidual, dict]:
    torch.manual_seed(seed)
    classes, ytr = _class_targets(y_train)
    lookup = {int(c): i for i, c in enumerate(classes.tolist())}
    yva = np.asarray([lookup[int(v)] for v in y_val], dtype=np.int64)
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = LabelEmbeddingResidual(base_train.shape[1], rank, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    bt = torch.from_numpy(base_train).float().to(device)
    bv = torch.from_numpy(base_val).float().to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    yv = torch.from_numpy(yva).long().to(device)
    if sample_weights is None:
        sample_weights = np.ones(len(y_train), dtype=np.float32)
    if len(sample_weights) != len(y_train):
        raise ValueError("sample_weights must align with label-embedding training rows")
    sample_weights_t = torch.from_numpy(
        np.asarray(sample_weights, dtype=np.float32)
    ).to(device)
    best_state = None
    best = float("inf")
    stale = 0
    for _ in range(settings.epochs):
        model.train()
        perm = torch.randperm(len(bt), device=device)
        for start in range(0, len(bt), settings.batch_size):
            idx = perm[start:start + settings.batch_size]
            logits, z, fine, _ = model(bt[idx], yt[idx], 1.0)
            per_sample = F.cross_entropy(logits, yt[idx], reduction="none")
            loss = (per_sample * sample_weights_t[idx]).sum() / sample_weights_t[idx].sum().clamp_min(1e-6)
            loss = loss + settings.fine_l2_weight * fine.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(bv, yv, 1.0)
            value = float(F.cross_entropy(logits, yv))
        if value < best - 1e-6:
            best = value
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError("label embedding training failed")
    if settings.restore_best:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "classes": classes.tolist(), "best_val_loss": best,
        "trainable_parameters": parameter_count(model),
        "checkpoint_selection": "best_validation" if settings.restore_best else "fixed_final_epoch",
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
        "sample_weight_mean": float(np.mean(sample_weights)),
    }


@torch.no_grad()
def encode_method(
    model: MatchedTokenMethod,
    base: np.ndarray,
    spec: np.ndarray,
    dosage: float,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    zs, fines, bases = [], [], []
    for start in range(0, len(base), batch_size):
        b = torch.from_numpy(base[start:start + batch_size]).float().to(device)
        s = torch.from_numpy(spec[start:start + batch_size]).float().to(device)
        z, fine, base_op = model.representation(b, s, dosage)
        zs.append(z.cpu().numpy().astype(np.float32))
        fines.append(fine.cpu().numpy().astype(np.float32))
        bases.append(base_op.cpu().numpy().astype(np.float32))
    return np.concatenate(zs), np.concatenate(fines), np.concatenate(bases)


@torch.no_grad()
def encode_label_embedding(
    model: LabelEmbeddingResidual,
    base: np.ndarray,
    labels_index: np.ndarray,
    dosage: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    b = torch.from_numpy(base).float().to(device)
    y = torch.from_numpy(labels_index).long().to(device)
    z, fine, base_op = model.representation(b, y, dosage)
    return z.cpu().numpy().astype(np.float32), fine.cpu().numpy().astype(np.float32), base_op.cpu().numpy().astype(np.float32)


def fixed_spectral_projection(spec: np.ndarray, token_dim: int, seed: int = 20260710) -> np.ndarray:
    """Label-free isometry from spectral dimension into the common token dimension."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(token_dim, spec.shape[1])).astype(np.float64)
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return (spec.astype(np.float32) @ q.T.astype(np.float32)).astype(np.float32)
