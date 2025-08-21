from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd

KEEP = ["run", "pass", "punt", "field_goal", "spike", "kneel"]


def norm_pt(pt):
    pt = str(pt) if pd.notna(pt) else None
    if pt in ("qb_spike", "spike"):
        return "spike"
    if pt in ("qb_kneel", "kneel"):
        return "kneel"
    if pt and pt.startswith("field_goal"):
        return "field_goal"
    return pt


def stats(x):
    x = x.dropna()
    if len(x) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, p10=np.nan, p50=np.nan, p90=np.nan)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        std=float(x.std()),
        p10=float(x.quantile(0.1)),
        p50=float(x.quantile(0.5)),
        p90=float(x.quantile(0.9)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="")
    ap.add_argument("--real", default="data/processed/plays_canonical.parquet")
    args = ap.parse_args()

    sim_path = args.sims or (
        sorted(glob.glob("runs/sim_*games.parquet"))[-1]
        if glob.glob("runs/sim_*games.parquet")
        else ""
    )
    if not sim_path:
        print("No sim parquet found in runs/ (expected runs/sim_*games.parquet)")
        sys.exit(1)

    print(f"\n== Quick Eval ==\nSIMS: {sim_path}\nREAL: {args.real}\n")
    sys.stdout.flush()

    sims = pd.read_parquet(sim_path)
    real = pd.read_parquet(args.real)

    real["play_type"] = real["play_type"].map(norm_pt)
    real = real[real["play_type"].isin(KEEP)]

    # Play-type share
    pt = pd.DataFrame(
        {
            "real": real["play_type"].value_counts(normalize=True).reindex(KEEP).fillna(0),
            "sim": sims["play_type"].value_counts(normalize=True).reindex(KEEP).fillna(0),
        }
    )
    pt["abs_diff"] = (pt["sim"] - pt["real"]).abs()

    print("-- Play-type share (real vs sim) --")
    print(pt.to_string())
    print("max_abs_diff:", round(pt["abs_diff"].max(), 3), "\n")
    sys.stdout.flush()

    # Situational pass rates
    def pass_rate(df, down, lo, hi):
        d = df[(df["down"] == down) & (df["ydstogo"].between(lo, hi))]
        return (d["play_type"] == "pass").mean() if len(d) else np.nan, len(d)

    r1, n1 = pass_rate(real, 1, 10, 10)
    s1, m1 = pass_rate(sims, 1, 10, 10)
    r3, n3 = pass_rate(real, 3, 7, 50)
    s3, m3 = pass_rate(sims, 3, 7, 50)

    print("-- Situational pass rates --")
    print(f"1st&10 pass rate   real={r1:.3f} (n={n1})  sim={s1:.3f} (n={m1})")
    print(f"3rd&7+ pass rate   real={r3:.3f} (n={n3})  sim={s3:.3f} (n={m3})\n")
    sys.stdout.flush()

    # Yards/play
    r = real.loc[real["play_type"].isin(["run", "pass"]), "yards_gained"].clip(-10, 80)
    s = (
        sims.loc[sims["play_type"].isin(["run", "pass"]), "yards"].clip(-10, 80)
        if "yards" in sims
        else pd.Series(dtype=float)
    )

    print("-- Yards/play (run+pass, clipped [-10,80]) --")
    from pprint import pprint

    print("real:")
    pprint(
        {
            k: (round(v, 3) if isinstance(v, int | float | np.floating) else v)
            for k, v in stats(r).items()
        }
    )
    print("sim: ")
    pprint(
        {
            k: (round(v, 3) if isinstance(v, int | float | np.floating) else v)
            for k, v in stats(s).items()
        }
    )
    print()
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("quick_eval error:", e)
        traceback.print_exc()
        sys.exit(1)
