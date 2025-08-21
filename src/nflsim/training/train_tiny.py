# mypy: ignore-errors
from __future__ import annotations
from nflsim.constants import (THIRD_AND_LONG_YTG, SECOND_AND_LONG_YTG, FIRST_AND_TEN_YTG, SHORT_YTG, SPIKE_CUTOFF_S, RED_ZONE_YARD, GOAL_TO_GO_YARD, RUN_PLAY_SECONDS, PASS_PLAY_SECONDS)

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from nflsim.config import FullConfig
from nflsim.models.play_transformer import PlayTransformer

PLAY_TYPES = ["run", "pass", "punt", "field_goal", "spike", "kneel"]
PT2IDX = {p: i for i, p in enumerate(PLAY_TYPES)}
RUN_IDX, PASS_IDX = PT2IDX["run"], PT2IDX["pass"]
PUNT_IDX, FG_IDX = PT2IDX["punt"], PT2IDX["field_goal"]
QUANTS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def yard_bucket(y):
    return int(np.clip(y, 0, 100))


def dist_bucket(d):
    d = int(max(1, min(99, int(d))))
    return d if d <= 20 else 21


def time_bucket(gs):
    return int(np.clip(int(gs) // 30, 0, 120))


def score_bucket(sd):
    sd = int(np.clip(int(sd), -28, 28))
    return (sd + 28) // 2


def flag_bucket(row):
    s = int((row.get("shotgun", 0) or 0) > 0)
    n = int((row.get("no_huddle", 0) or 0) > 0)
    return s * 2 + n


def temp_bucket(t):
    t = 70 if pd.isna(t) else float(t)
    bins = [-100, 32, 50, 65, 80, 95, 1000]
    return int(np.digitize([t], bins)[0] - 1)  # 0..5


def wind_bucket(w):
    w = 0 if pd.isna(w) else float(w)
    bins = [-1, 5, 10, 15, 25, 1000]
    return int(np.digitize([w], bins)[0] - 1)  # 0..4


def bool_bucket(b):
    return 1 if bool(b) else 0


def pinball_loss(pred_q, y, quantiles):
    y = y.unsqueeze(1).expand_as(pred_q)
    e = y - pred_q
    qs = torch.tensor(quantiles, device=pred_q.device).unsqueeze(0)
    return torch.maximum(qs * e, (qs - 1) * e).mean()


class PlaysDataset(Dataset):
    def __init__(self, parquet_path, sample_plays=None, context_k=8):
        df = pd.read_parquet(parquet_path)

        if sample_plays is not None and len(df) > sample_plays:
            df = df.sample(sample_plays, random_state=123).sort_values(["game_id", "play_id"])

        # Normalize labels
        df["play_type_norm"] = (
            df["play_type"].astype(str).replace({"qb_spike": "spike", "qb_kneel": "kneel"})
        )
        df = df[df["play_type_norm"].isin(PLAY_TYPES)].copy()

        # Base fields
        df["flags_b"] = df.apply(flag_bucket, axis=1)
        df["ydstogo"] = df["ydstogo"].clip(1, 99)
        df["yardline_100"] = df["yardline_100"].clip(0, 100)
        df["game_seconds_remaining"] = df["game_seconds_remaining"].clip(lower=0)
        df["score_differential"] = df["score_differential"].fillna(0).clip(-60, 60)
        df["yards_gained"] = df["yards_gained"].fillna(0.0).clip(-10, 80)

        # Categorical codes
        for col in ["posteam", "defteam", "season"]:
            if col not in df:
                df[col] = "UNK"
            df[col] = pd.Categorical(df[col])
            df[f"{col}_code"] = df[col].cat.codes.astype("int64")

        # Environment (fill if missing)
        if "roof" not in df:
            df["roof"] = np.full(len(df), "outdoors")
        if "surface" not in df:
            df["surface"] = np.full(len(df), "grass")
        df["temp"] = df.get("temp", pd.Series(np.full(len(df), 70.0))).astype(float)
        df["wind"] = df.get("wind", pd.Series(np.full(len(df), 5.0))).astype(float)

        df["roof"] = pd.Categorical(df["roof"].fillna("outdoors"))
        df["surface"] = pd.Categorical(df["surface"].fillna("grass"))
        df["roof_code"] = df["roof"].cat.codes.astype("int64")
        df["surface_code"] = df["surface"].cat.codes.astype("int64")
        df["temp_b"] = df["temp"].apply(temp_bucket)
        df["wind_b"] = df["wind"].apply(wind_bucket)

        # Context flags
        df["g2g_flag"] = (
            (df["ydstogo"] == df["yardline_100"]) & (df["yardline_100"] <= 10)
        ).astype(int)
        df["rz_flag"] = (df["yardline_100"] <= 20).astype(int)

        self.df = df.reset_index(drop=True)
        self.k = context_k

        # cardinalities for model construction + vocab save
        self.team_card = int(max(df["posteam_code"].max(), df["defteam_code"].max()) + 1)
        self.season_card = int(df["season_code"].max() + 1)
        self.roof_card = int(df["roof_code"].max() + 1)
        self.surface_card = int(df["surface_code"].max() + 1)
        self.vocabs = {
            "posteam": df["posteam"].cat.categories.tolist(),
            "defteam": df["defteam"].cat.categories.tolist(),
            "season": df["season"].cat.categories.tolist(),
            "roof": df["roof"].cat.categories.tolist(),
            "surface": df["surface"].cat.categories.tolist(),
        }

    def __len__(self):
        return max(0, len(self.df) - self.k - 1)

    def __getitem__(self, idx):
        ctx = self.df.iloc[idx : idx + self.k]
        nxt = self.df.iloc[idx + self.k]
        x = {
            "down": torch.tensor(ctx["down"].clip(1, 4).values - 1, dtype=torch.long),
            "yard_bucket": torch.tensor(
                ctx["yardline_100"].apply(yard_bucket).values, dtype=torch.long
            ),
            "dist_bucket": torch.tensor(ctx["ydstogo"].apply(dist_bucket).values, dtype=torch.long),
            "flags": torch.tensor(ctx["flags_b"].values, dtype=torch.long),
            "time_bucket": torch.tensor(
                ctx["game_seconds_remaining"].apply(time_bucket).values, dtype=torch.long
            ),
            "score_bucket": torch.tensor(
                ctx["score_differential"].apply(score_bucket).values, dtype=torch.long
            ),
            "posteam": torch.tensor(ctx["posteam_code"].values, dtype=torch.long),
            "defteam": torch.tensor(ctx["defteam_code"].values, dtype=torch.long),
            "season": torch.tensor(ctx["season_code"].values, dtype=torch.long),
            "roof_bucket": torch.tensor(ctx["roof_code"].values, dtype=torch.long),
            "surface_bucket": torch.tensor(ctx["surface_code"].values, dtype=torch.long),
            "temp_bucket": torch.tensor(ctx["temp_b"].values, dtype=torch.long),
            "wind_bucket": torch.tensor(ctx["wind_b"].values, dtype=torch.long),
            "g2g_flag": torch.tensor(ctx["g2g_flag"].values, dtype=torch.long),
            "rz_flag": torch.tensor(ctx["rz_flag"].values, dtype=torch.long),
        }
        y = {
            "pt": torch.tensor(PT2IDX[str(nxt["play_type_norm"])], dtype=torch.long),
            "yards": torch.tensor(float(nxt["yards_gained"]), dtype=torch.float32),
        }
        return x, y


def collate(batch):
    keys = [
        "down",
        "yard_bucket",
        "dist_bucket",
        "flags",
        "time_bucket",
        "score_bucket",
        "posteam",
        "defteam",
        "season",
        "roof_bucket",
        "surface_bucket",
        "temp_bucket",
        "wind_bucket",
        "g2g_flag",
        "rz_flag",
    ]
    xs = {k: [] for k in keys}
    ys_pt, ys_yards = [], []
    for x, y in batch:
        for k in keys:
            xs[k].append(x[k])
        ys_pt.append(y["pt"])
        ys_yards.append(y["yards"])
    out = {k: torch.stack(v, dim=0) for k, v in xs.items()}
    return out, torch.stack(ys_pt), torch.stack(ys_yards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--context_k", type=int, default=12)
    args = ap.parse_args()
    cfg = FullConfig(**yaml.safe_load(open(args.config)))
    set_seed(cfg.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = PlaysDataset(cfg.data.processed_path, cfg.data.sample_plays, context_k=args.context_k)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate)

    # Tempered class weights (inverse-freq^0.5)
    counts = np.bincount([PT2IDX[p] for p in ds.df["play_type_norm"]], minlength=len(PLAY_TYPES))
    cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=device)

    model = PlayTransformer(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        n_play_types=len(PLAY_TYPES),
        n_team=max(ds.team_card, 40),
        n_season=max(ds.season_card, 16),
        n_roof=max(ds.roof_card, 5),
        n_surface=max(ds.surface_card, 6),
        max_len=args.context_k,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95)
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.max_steps)
    warmup = 800

    steps = 0
    model.train()
    for _epoch in range(999):
        for xb, y_pt, y_yards in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            y_pt = y_pt.to(device)
            y_yards = y_yards.to(device)

            # AMP on MPS not available; use full precision for stability
            out = model(xb)

            # sample weights: 3rd&7+, last 2 minutes
            last_down = xb["down"][:, -1]  # 0=1st, 1=2nd, 2=3rd, 3=4th
            last_dist = xb["dist_bucket"][:, -1]
            last_time = xb["time_bucket"][:, -1]
            is_3rd7 = (last_down == 2) & (last_dist >= 7)
            is_2min = last_time <= 4
            wt = torch.ones_like(y_pt, dtype=torch.float32, device=device)
            wt = wt * torch.where(
                is_3rd7, torch.tensor(2.0, device=device), torch.tensor(1.0, device=device)
            )
            wt = wt * torch.where(
                is_2min, torch.tensor(1.3, device=device), torch.tensor(1.0, device=device)
            )

            # play-type loss
            ce = F.cross_entropy(
                out["play_type_logits"], y_pt, weight=cw, label_smoothing=0.03, reduction="none"
            )
            loss_pt = (ce * wt).mean()

            # yards quantile loss on run/pass only + smoothness regularizer
            yards_q_run = out["yards_q_run"]
            yards_q_pass = out["yards_q_pass"]
            mask_run = y_pt == RUN_IDX
            mask_pass = y_pt == PASS_IDX
            mask_rp = mask_run | mask_pass
            if mask_rp.any():
                sel_q = torch.zeros_like(yards_q_run)
                sel_q[mask_run] = yards_q_run[mask_run]
                sel_q[mask_pass] = yards_q_pass[mask_pass]
                loss_y = pinball_loss(sel_q[mask_rp], y_yards[mask_rp], QUANTS)
                gaps = sel_q[:, 1:] - sel_q[:, :-1]
                reg = 1e-3 * (gaps**2).mean()
            else:
                loss_y, reg = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            # Aux heads on 4th down (attempt logits)
            m4 = last_down == 3  # 0-indexed -> 3 == 4th
            if m4.any():
                fg_target = (y_pt == FG_IDX).float()
                punt_target = (y_pt == PUNT_IDX).float()
                loss_fg = F.binary_cross_entropy_with_logits(
                    out["fg_attempt_logit"][m4], fg_target[m4]
                )
                loss_punt = F.binary_cross_entropy_with_logits(
                    out["punt_attempt_logit"][m4], punt_target[m4]
                )
            else:
                loss_fg = loss_punt = torch.tensor(0.0, device=device)

            loss = loss_pt + 0.25 * loss_y + reg + 0.3 * (loss_fg + loss_punt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if steps > warmup:
                sch.step()
            if steps % 100 == 0:
                print(
                    f"step {steps} | loss_pt {loss_pt.item():.4f} | loss_y {loss_y.item():.4f} | aux {(loss_fg+loss_punt).item():.4f}"
                )
            if steps >= cfg.train.max_steps:
                break
        if steps >= cfg.train.max_steps:
            break

    os.makedirs("runs", exist_ok=True)
    # persist vocab so simulate uses exact same ids
    json.dump(ds.vocabs, open("runs/vocab.json", "w"))
    torch.save({"model_state": model.state_dict()}, "runs/tiny.ckpt")
    print("saved runs/tiny.ckpt and runs/vocab.json")


if __name__ == "__main__":
    main()
