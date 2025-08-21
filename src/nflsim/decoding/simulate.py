# mypy: ignore-errors
from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nflsim.models.play_transformer import PlayTransformer
from nflsim.rules.fsm import State, clamp_state, legal_play_types

PLAY_TYPES = ["run", "pass", "punt", "field_goal", "spike", "kneel"]
RUN, PASS, PUNT, FG, SPIKE, KNEEL = range(6)


def top_p_sample(logits, legal_set, top_p=0.85):
    mask = torch.full_like(logits, -1e9)
    for i, pt in enumerate(PLAY_TYPES):
        if pt in legal_set:
            mask[i] = 0.0
    logits = logits + mask
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    idx = np.argsort(probs)[::-1]
    cum = probs[idx].cumsum()
    keep = idx[cum <= top_p]
    if len(keep) == 0:
        keep = idx[:2]
    probs = probs[keep] / probs[keep].sum()
    return int(np.random.choice(keep, p=probs))


def fg_make_prob(kd: float) -> float:
    # logistic by kick distance (LOS + 17)
    return 1.0 / (1.0 + np.exp((kd - 45.0) / 3.5))


def punt_bias(yl100: int) -> float:
    # mild physics prior: punt more in own territory
    if yl100 >= 80:
        return 0.7
    if yl100 >= 70:
        return 0.4
    if yl100 >= 60:
        return 0.2
    return 0.0


def sample_from_quantiles(q):
    q = np.sort(np.asarray(q, dtype=float))
    u = np.random.rand()
    k = min(int(u * 9), 8)
    if k == 0:
        lo, hi = -10.0, q[0]
    elif k == 8:
        lo, hi = q[8], 80.0
    else:
        lo, hi = q[k - 1], q[k]
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-3:
        return float(np.clip(hi, -10, 80))
    return float(np.clip(np.random.uniform(lo, hi), -10, 80))


def time_bucket(gs):
    return int(np.clip(gs // 30, 0, 120))


def score_bucket(sd):
    sd = int(np.clip(sd, -28, 28))
    return (sd + 28) // 2


def choose_flags(down: int, ytg: int, gs: int):
    shotgun_p = 0.7 if (down == 3 and ytg >= 7) else (0.55 if ytg >= 8 else 0.35)
    if gs <= 120:
        shotgun_p = max(shotgun_p, 0.6)
    no_huddle_p = 0.4 if gs <= 120 else 0.1
    shotgun = int(np.random.rand() < shotgun_p)
    nohuddle = int(np.random.rand() < no_huddle_p)
    return shotgun * 2 + nohuddle


def flip_possession(s: State, new_ball_yardline_old_offense: int):
    s.possession = "away" if s.possession == "home" else "home"
    s.yardline_100 = int(np.clip(100 - new_ball_yardline_old_offense, 0, 100))
    s.down, s.ydstogo = 1, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--checkpoint", type=str, default="runs/tiny.ckpt")
    ap.add_argument("--out", type=str, default="runs/sim_games.parquet")
    ap.add_argument("--top_p", type=float, default=0.85)
    ap.add_argument("--context_k", type=int, default=12)
    ap.add_argument("--season_code", type=int, default=0)
    ap.add_argument("--home_team_code", type=int, default=0)
    ap.add_argument("--away_team_code", type=int, default=1)
    ap.add_argument("--roof_code", type=int, default=0)  # 0: first in vocab (likely 'outdoors')
    ap.add_argument("--surface_code", type=int, default=0)  # e.g., 'grass'
    ap.add_argument("--temp_bucket", type=int, default=3)  # ~65-80F
    ap.add_argument("--wind_bucket", type=int, default=1)  # 0-5 or 6-10 mph
    args = ap.parse_args()

    # load vocab (if exists) to keep ids consistent
    try:
        json.load(open("runs/vocab.json"))
        # we'll just trust provided codes or 0/1
                vocab primarily prevents train/sim drift
    except Exception:
        pass

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PlayTransformer()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    for g in range(args.n_games):
        s = State(
            qtr=1,
            game_seconds_remaining=3600,
            down=1,
            ydstogo=10,
            yardline_100=75,
            posteam_timeouts=3,
            defteam_timeouts=3,
            possession="home",
            score_home=0,
            score_away=0,
        )
        drives = 0
        K = args.context_k

        def cur_teams():
            return (
                (args.home_team_code, args.away_team_code)
                if s.possession == "home"
                else (args.away_team_code, args.home_team_code)
            )

        down_buf = deque([s.down - 1] * K, maxlen=K)
        dist_buf = deque([min(21, max(1, s.ydstogo))] * K, maxlen=K)
        yard_buf = deque([int(np.clip(s.yardline_100, 0, 100))] * K, maxlen=K)
        time_buf = deque([time_bucket(s.game_seconds_remaining)] * K, maxlen=K)
        score_buf = deque([score_bucket(s.score_home - s.score_away)] * K, maxlen=K)
        flags_buf = deque([choose_flags(s.down, s.ydstogo, s.game_seconds_remaining)] * K, maxlen=K)
        posteam_buf = deque([cur_teams()[0]] * K, maxlen=K)
        defteam_buf = deque([cur_teams()[1]] * K, maxlen=K)
        season_buf = deque([args.season_code] * K, maxlen=K)
        roof_buf = deque([args.roof_code] * K, maxlen=K)
        surface_buf = deque([args.surface_code] * K, maxlen=K)
        temp_buf = deque([args.temp_bucket] * K, maxlen=K)
        wind_buf = deque([args.wind_bucket] * K, maxlen=K)
        g2g_buf = deque([0] * K, maxlen=K)
        rz_buf = deque([int(s.yardline_100 <= 20)] * K, maxlen=K)

        while s.game_seconds_remaining > 0 and drives < 24:
            legal = legal_play_types(s)
            xb = {
                "down": torch.tensor([list(down_buf)], dtype=torch.long),
                "dist_bucket": torch.tensor([list(dist_buf)], dtype=torch.long),
                "yard_bucket": torch.tensor([list(yard_buf)], dtype=torch.long),
                "time_bucket": torch.tensor([list(time_buf)], dtype=torch.long),
                "score_bucket": torch.tensor([list(score_buf)], dtype=torch.long),
                "flags": torch.tensor([list(flags_buf)], dtype=torch.long),
                "posteam": torch.tensor([list(posteam_buf)], dtype=torch.long),
                "defteam": torch.tensor([list(defteam_buf)], dtype=torch.long),
                "season": torch.tensor([list(season_buf)], dtype=torch.long),
                "roof_bucket": torch.tensor([list(roof_buf)], dtype=torch.long),
                "surface_bucket": torch.tensor([list(surface_buf)], dtype=torch.long),
                "temp_bucket": torch.tensor([list(temp_buf)], dtype=torch.long),
                "wind_bucket": torch.tensor([list(wind_buf)], dtype=torch.long),
                "g2g_flag": torch.tensor([list(g2g_buf)], dtype=torch.long),
                "rz_flag": torch.tensor([list(rz_buf)], dtype=torch.long),
            }
            with torch.no_grad():
                out = model(xb)

            logits = out["play_type_logits"].view(-1)

            # learned special-teams preferences on 4th down
            if s.down == 4:
                logits[FG] += float(out["fg_attempt_logit"])
                logits[PUNT] += float(out["punt_attempt_logit"])
                logits[SPIKE] -= 2.0
                # small physics prior
                logits[PUNT] += punt_bias(s.yardline_100)

            # light pass/run nudges by situation
            if s.down == 3 and s.ydstogo >= 7:
                logits[PASS] += 1.0
                logits[RUN] -= 0.3
            if s.down == 2 and s.ydstogo >= 8:
                logits[PASS] += 0.3
            if s.down == 1 and s.ydstogo == 10:
                logits[PASS] += 0.2
            if s.ydstogo <= 2:
                logits[RUN] += 0.2
            if s.game_seconds_remaining > 60:
                logits[SPIKE] -= 1.0

            pt_idx = top_p_sample(logits, legal, args.top_p)
            play_type = PLAY_TYPES[pt_idx]

            # yards (two-headed)
            if play_type == "run":
                yards = sample_from_quantiles(out["yards_q_run"].view(-1).cpu().numpy())
            elif play_type == "pass":
                yards = sample_from_quantiles(out["yards_q_pass"].view(-1).cpu().numpy())
            else:
                yards = 0.0

            # state update
            if play_type in ("run", "pass"):
                s.yardline_100 = max(0, s.yardline_100 - yards)
                if yards >= s.ydstogo:
                    s.down, s.ydstogo = 1, 10
                else:
                    s.down = min(4, s.down + 1)
                    s.ydstogo = max(1, s.ydstogo - yards)
            elif play_type == "punt":
                new_yard_old_offense = s.yardline_100 + 42
                flip_possession(s, new_yard_old_offense)
                drives += 1
            elif play_type == "field_goal":
                kd = s.yardline_100 + 17
                if np.random.rand() < fg_make_prob(kd):
                    if s.possession == "home":
                        s.score_home += 3
                    else:
                        s.score_away += 3
                    flip_possession(s, 25)
                else:
                    flip_possession(s, s.yardline_100)
                drives += 1
            elif play_type in ("spike", "kneel"):
                s.down = min(4, s.down + 1)

            # clock & clamp
            s.game_seconds_remaining = max(
                0, s.game_seconds_remaining - (38 if play_type in ("run", "kneel") else 28)
            )
            s = clamp_state(s)

            # roll context forward
            down_buf.append(s.down - 1)
            dist_buf.append(min(21, max(1, s.ydstogo)))
            yard_buf.append(int(np.clip(s.yardline_100, 0, 100)))
            time_buf.append(time_bucket(s.game_seconds_remaining))
            score_buf.append(score_bucket(s.score_home - s.score_away))
            flags_buf.append(choose_flags(s.down, s.ydstogo, s.game_seconds_remaining))
            p, d = cur_teams()
            posteam_buf.append(p)
            defteam_buf.append(d)
            season_buf.append(season_buf[-1])
            roof_buf.append(roof_buf[-1])
            surface_buf.append(surface_buf[-1])
            temp_buf.append(temp_buf[-1])
            wind_buf.append(wind_buf[-1])
            g2g_buf.append(int((s.yardline_100 <= 10) and (s.ydstogo >= s.yardline_100)))
            rz_buf.append(int(s.yardline_100 <= 20))

            rows.append(
                {
                    "game": g,
                    "qtr": s.qtr,
                    "secs": s.game_seconds_remaining,
                    "down": s.down,
                    "ydstogo": s.ydstogo,
                    "yardline_100": s.yardline_100,
                    "play_type": play_type,
                    "yards": float(yards),
                    "score_home": s.score_home,
                    "score_away": s.score_away,
                }
            )

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
