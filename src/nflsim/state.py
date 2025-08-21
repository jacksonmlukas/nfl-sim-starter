from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GameState:
    season: int
    quarter: int            # 1..5 (OT=5)
    clock_bin: int          # 0..719 (5s bins within quarter, counting down)
    offense: str
    defense: str
    down: int               # 1..4
    distance: int           # 1..20, 21=20+
    yardline: int           # 0..100 offense->opp
    to_off: int             # 0..3
    to_def: int             # 0..3
    score_off: int
    score_def: int
    possession_index: int   # increments on change of possession
