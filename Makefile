# Convenience commands
.PHONY: setup data test train simulate report
setup:
	conda create -n nflsim python=3.11 -y && conda activate nflsim && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e . && pip install -r requirements-dev.txt
data:
	python scripts/prepare_data.py --seasons 2019 2020 2021 2022 2023 2024
test:
	pytest -q
train:
	python scripts/train_tiny.py --config configs/tiny.yaml
simulate:
	python scripts/simulate_games.py --n_games 50 --checkpoint runs/tiny.ckpt --out runs/sim_50_games.parquet
report:
	python scripts/eval_report.py --sims runs/sim_50_games.parquet --real data/processed/plays_canonical.parquet --out runs/report.md
