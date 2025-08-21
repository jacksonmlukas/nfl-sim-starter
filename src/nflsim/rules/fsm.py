from __future__ import annotations
import numpy as np
from dataclasses import replace
from nflsim.state import GameState

PLAY_TYPES = ("run","pass","punt","fg","spike","kneel","ko","onside","penalty")

class RulesFSM:
    def legal_actions(self, s: GameState) -> dict[str, np.ndarray]:
        mask_pt = np.zeros(len(PLAY_TYPES), dtype=bool)
        if s.down == 4 and s.distance > 1:
            mask_pt[[0,1,2,3]] = True  # run, pass, punt, fg
        else:
            mask_pt[[0,1,4,5]] = True  # run, pass, spike, kneel
        return {"play_type": mask_pt}

    def apply_outcome(self, s: GameState, *, yards: int, turnover: bool,
                      score_delta: int, clock_off: int) -> GameState:
        yl = max(0, min(100, s.yardline + yards))
        made_line = yards >= s.distance
        if turnover:
            ns = replace(s, offense=s.defense, defense=s.offense,
                         possession_index=s.possession_index + 1,
                         down=1, distance=10, yardline=100-yl)
        else:
            if made_line:
                ns = replace(s, down=1, distance=10, yardline=yl)
            else:
                ns = replace(s, down=min(4, s.down + 1),
                             distance=max(1, s.distance - yards), yardline=yl)
        nb = max(0, s.clock_bin - clock_off)
        ns = replace(ns, clock_bin=nb, score_off=s.score_off + score_delta)
        if ns.clock_bin < 0:
            raise ValueError("Clock went negative")
        return ns
