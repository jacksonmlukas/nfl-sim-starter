from __future__ import annotations
from typing import Tuple
import numpy as np

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Binary Brier score (mean squared error)."""
    probs = probs.astype(np.float64)
    labels = labels.astype(np.float64)
    return float(np.mean((probs - labels) ** 2))

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (binary)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(np.mean(probs[m]))
        acc = float(np.mean(labels[m]))
        ece += abs(acc - conf) * (np.mean(m))
    return float(ece)

def pit_histogram(samples: np.ndarray, cdf_vals: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability Integral Transform. `samples` are target values; `cdf_vals` are model CDF at each sample.
    Return (hist, bin_edges).
    """
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    hist, edges = np.histogram(cdf_vals, bins=n_bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float64), edges
