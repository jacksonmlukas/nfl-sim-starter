import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicQuantileHead(nn.Module):
    """
    Produces strictly increasing quantiles by parameterizing positive gaps.
    Returns q10..q90 as a [B, n_q] tensor.
    """

    def __init__(self, d_model: int, n_q: int = 9):
        super().__init__()
        self.base = nn.Linear(d_model, 1)  # anchor (close to median)
        self.gaps = nn.Linear(d_model, n_q)  # positive gaps -> cumulatively increasing

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        base = self.base(h)  # [B,1]
        gaps = F.softplus(self.gaps(h))  # [B,n_q]  >= 0
        q = torch.cumsum(gaps, dim=-1)  # monotone increasing
        q = q - q.mean(dim=-1, keepdim=True) + base  # center around base
        return q
