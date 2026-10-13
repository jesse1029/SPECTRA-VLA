"""Action-sensitive policy subspace (section 4.3).

Given a batch of action-sensitivity gradients of the frozen VLA w.r.t. its visual
tokens, fit a low-rank basis U_pol whose span captures the directions the frozen
policy uses to decide actions.  P_pol = U_pol U_pol^T ; P_perp = I - P_pol.

The basis lives in the per-token feature space (dim D), shared across the N token
positions, so it applies to any [B, N, D] token tensor.
"""
import torch
import torch.nn.functional as F


class PolicySubspace:
    def __init__(self, rank: int = 64):
        self.rank = rank
        self.U = None  # [D, rank]

    def fit(self, gradients: torch.Tensor):
        """gradients: [M, N, D] action-sensitivity per probe. Fit a D x rank basis.

        We treat every (probe, token) pair as a sample of an action-sensitive
        direction in feature space R^D and take the top-`rank` right singular
        vectors of the stacked [M*N, D] matrix.
        """
        M, N, D = gradients.shape
        G = gradients.reshape(M * N, D).float()
        G = F.normalize(G, dim=-1)
        # randomized/truncated SVD via torch.svd_lowrank for efficiency
        q = min(self.rank + 16, min(G.shape))
        U_, S_, V_ = torch.svd_lowrank(G, q=q)
        self.U = V_[:, : self.rank].contiguous()  # [D, rank]
        self.singular_values = S_[: self.rank].detach().cpu()
        return self

    def to(self, device):
        if self.U is not None:
            self.U = self.U.to(device)
        return self

    def project(self, z):
        """z: [B, N, D] -> projection onto policy subspace, same shape."""
        U = self.U.to(z.dtype).to(z.device)
        coeff = z @ U          # [B, N, rank]
        return coeff @ U.T     # [B, N, D]

    def residual(self, z):
        return z - self.project(z)

    def state_dict(self):
        return {"U": self.U.detach().cpu(), "rank": self.rank}

    def load_state_dict(self, sd):
        self.U = sd["U"]
        self.rank = sd["rank"]
        return self
