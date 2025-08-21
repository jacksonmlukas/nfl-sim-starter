import numpy as np


def apply_masks(
    logits: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    For each head, set illegal-logit positions to -inf based on boolean masks.
    """
    out = {}
    for head, lg in logits.items():
        m = masks.get(head)
        if m is None:
            out[head] = lg
            continue
        lg = lg.copy()
        lg[~m] = -np.inf
        out[head] = lg
    return out
