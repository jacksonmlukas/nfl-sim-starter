# NFL Play-by-Play Transformer Simulator — Starter

This repo is a **free-only** starter to prove feasibility on a MacBook (Apple Silicon or Intel) before paying for any GPU time. It matches the plan we discussed: rules-aware constrained decoding, a tiny transformer, and basic calibration + evaluation.

## 0) Prereqs (Mac)
- macOS 12+ recommended
- Xcode Command Line Tools: `xcode-select --install`
- Miniforge (recommended) or conda: https://github.com/conda-forge/miniforge

## 1) Install (new env)
```bash
conda create -n nflsim python=3.11 -y
conda activate nflsim
# Install PyTorch with Metal (MPS) support (on Apple Silicon/intel macs it's the same pip)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Core deps
pip install -e .
# (Optional) Dev extras
pip install -r requirements-dev.txt
```

> Note: We pin CPU wheels by default to keep it universal. On Apple Silicon you still get MPS acceleration automatically when available.

### Verify PyTorch + MPS
```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("MPS available:", mps)
PY
```

## 2) Pull data (locally, free)
```bash
python scripts/prepare_data.py --seasons 2019 2020 2021 2022 2023 2024
```
This downloads nflverse/nflfastR play-by-play to `data/raw/` and writes **canonicalized Parquet** to `data/processed/plays_canonical.parquet` with a strict **feature whitelist** (no leakage).

## 3) Run tests (rules + leakage guards + smoke sim)
```bash
pytest -q
```

## 4) Train a tiny prototype (Mac-friendly)
```bash
python scripts/train_tiny.py --config configs/tiny.yaml
```
This trains a **small transformer** (4 layers, d=256) with scheduled sampling and constrained decoding on a sample of ~50–100k plays.

## 5) Simulate games
```bash
python scripts/simulate_games.py --n_games 50 --checkpoint runs/tiny.ckpt --out runs/sim_50_games.parquet
```

## 6) Evaluate calibration + realism
```bash
python scripts/eval_report.py --sims runs/sim_50_games.parquet --real data/processed/plays_canonical.parquet --out runs/report.md
```

## 7) Next steps (still free)
- Expand the rules FSM (timeouts, OT variants, penalties).
- Increase context from K=6 to K=8 once stable.
- Add batch-level moment matching on run/pass % and yards/play in `train_tiny.py`.

When you’re ready to scale, swap configs to a larger model and train on a cloud 4090/A100 for cheap. This starter keeps that path open.
