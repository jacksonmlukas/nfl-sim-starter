from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PT_ORDER = ["run","pass","punt","field_goal","spike","kneel"]

def norm_pt(pt:str):
    if pd.isna(pt): return None
    pt = str(pt)
    if pt in ("qb_spike","spike"): return "spike"
    if pt in ("qb_kneel","kneel"): return "kneel"
    if pt.startswith("field_goal"): return "field_goal"
    return pt

def situational_slice(df: pd.DataFrame, down:int, ytg_min:int, ytg_max:int):
    m = (df["down"]==down) & (df["ydstogo"].between(ytg_min, ytg_max, inclusive="both"))
    return df.loc[m].copy()

def plot_hist_overlay(real, sim, bins, title, out_png):
    plt.figure(figsize=(6,4))
    plt.hist(real, bins=bins, alpha=0.5, density=True, label="real")
    plt.hist(sim, bins=bins, alpha=0.5, density=True, label="sim")
    plt.xlabel("yards"); plt.ylabel("density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--out', default='runs/report.md')
    args = ap.parse_args()

    out_dir = Path(args.out).parent; out_dir.mkdir(parents=True, exist_ok=True)
    sims = pd.read_parquet(args.sims)
    real = pd.read_parquet(args.real)

    real["play_type_norm"] = real["play_type"].map(norm_pt)
    real = real[real["play_type_norm"].isin(PT_ORDER)].copy()

    # 1) Play-type distribution
    share_real = real["play_type_norm"].value_counts(normalize=True).reindex(PT_ORDER).fillna(0)
    share_sim  = sims["play_type"].value_counts(normalize=True).reindex(PT_ORDER).fillna(0)
    dist_tbl = pd.DataFrame({"real": share_real.round(3), "sim": share_sim.round(3)})

    # 2) Yards/play overlay (run+pass only)
    rp_real = real.loc[real["play_type_norm"].isin(["run","pass"]), "yards_gained"].clip(-10,80)
    rp_sim  = sims.loc[sims["play_type"].isin(["run","pass"]), "yards"].clip(-10,80) if "yards" in sims else pd.Series(dtype=float)
    yards_png = out_dir / "yards_overlay.png"
    if len(rp_sim) > 0 and len(rp_real) > 0:
        plot_hist_overlay(rp_real, rp_sim, bins=np.arange(-10,81,2), 
                          title="Yards/Play (Run+Pass, clipped [-10,80])", out_png=yards_png)

    # 3) Situational pass rate: 1st & 10; 3rd & 7+
    # Derive ydstogo in sims if missing (we save it in simulate patch)
    s_1st10 = situational_slice(sims, 1, 10, 10)
    r_1st10 = situational_slice(real.rename(columns={"play_type_norm":"play_type"}), 1, 10, 10)
    s_3rd7  = situational_slice(sims, 3, 7, 50)
    r_3rd7  = situational_slice(real.rename(columns={"play_type_norm":"play_type"}), 3, 7, 50)
    def pass_rate(df): 
        if len(df)==0: return np.nan
        return (df["play_type"]=="pass").mean()
    sit_tbl = pd.DataFrame({
        "situation": ["1st & 10","3rd & 7+"],
        "real_pass_rate": [pass_rate(r_1st10), pass_rate(r_3rd7)],
        "sim_pass_rate":  [pass_rate(s_1st10), pass_rate(s_3rd7)],
    }).round(3)

    # 4) Summary stats for yards
    def stats(x):
        x = x.dropna()
        if len(x)==0: return dict(n=0, mean=np.nan, std=np.nan, p10=np.nan, p50=np.nan, p90=np.nan)
        return dict(n=len(x), mean=x.mean(), std=x.std(), p10=x.quantile(.1), p50=x.quantile(.5), p90=x.quantile(.9))
    yard_stats = pd.DataFrame({"real": stats(rp_real), "sim": stats(rp_sim)}).round(2)

    # Write markdown report
    with open(args.out, "w") as f:
        f.write("# Simulation Evaluation Report\n\n")
        f.write(f"- Sims: **{len(sims):,}** plays from `{args.sims}`\n")
        f.write(f"- Real: sourced from `{args.real}`\n\n")

        f.write("## Play-type distribution\n\n")
        f.write(dist_tbl.to_string() + "\n\n")

        f.write("## Situational pass rates\n\n")
        f.write(sit_tbl.to_string(index=False) + "\n\n")

        if yards_png.exists():
            f.write("## Yards/Play Overlay (Run+Pass)\n\n")
            f.write(f"![Yards Overlay]({yards_png.name})\n\n")
            f.write(yard_stats.to_markdown() + "\n\n")
        else:
            f.write("## Yards/Play\n\n")
            f.write("_Sim data lacks `yards`; update simulate.py with the patch to enable._\n\n")

        f.write("## Notes\n")
        f.write("- Early prototype uses simple state updates and median-yards proxy; expect drift.\n")
        f.write("- Next step: train real targets (play_type + yards quantiles) and feed them into the decoder.\n")
    print("Wrote", args.out)
