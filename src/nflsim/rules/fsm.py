# Minimal football rules FSM (expand later)
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class State:
    qtr: int
    game_seconds_remaining: int
    down: int
    ydstogo: int
    yardline_100: int
    posteam_timeouts: int
    defteam_timeouts: int
    possession: str  # 'home'/'away'
    score_home: int
    score_away: int

PLAY_TYPES = ['run','pass','punt','field_goal','spike','kneel']

def legal_play_types(s: State) -> list[str]:
    legal = []
    # Basic offense options
    legal += ['run','pass']
    # 4th down: allow special teams
    if s.down == 4:
        # FG only if within ~60 yard FG (yardline_100 <= 43 from end zone)
        if s.yardline_100 <= 43:
            legal.append('field_goal')
        legal.append('punt')
    # End-game mechanics (very simplified)
    if s.game_seconds_remaining < 120 and s.down in (1,2,3) and s.yardline_100 < 90:
        legal.append('spike')
    if s.game_seconds_remaining < 120 and s.down in (1,2,3) and s.yardline_100 <= 5:
        legal.append('kneel')
    return sorted(set(legal))

def clamp_state(s: State) -> State:
    s.down = max(1, min(4, s.down))
    s.yardline_100 = max(0, min(100, s.yardline_100))
    s.ydstogo = max(1, min(99, s.ydstogo))
    s.game_seconds_remaining = max(0, s.game_seconds_remaining)
    return s
