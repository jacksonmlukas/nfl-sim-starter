from __future__ import annotations
import numpy as np

def shift_proe(logits: np.ndarray, pass_idx: int, run_idx: int, shift: float) -> None:
    """Increase/decrease pass-vs-run bias by adding an offset to pass and subtracting from run."""
    if 0 <= pass_idx < logits.shape[0]:
        logits[pass_idx] += shift
    if 0 <= run_idx < logits.shape[0]:
        logits[run_idx] -= shift

def scale_fourth_down(logits: np.ndarray, indices: dict[str, int], scale: float) -> None:
    """On 4th down, reduce punt/fg relative weight (scale<1 favors 'go'); or increase (scale>1)."""
    for k in ("punt", "fg"):
        idx = indices.get(k, -1)
        if 0 <= idx < logits.shape[0]:
            logits[idx] = logits[idx] / max(scale, 1e-6)
