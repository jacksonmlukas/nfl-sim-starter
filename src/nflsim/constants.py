from __future__ import annotations

# Situational thresholds
THIRD_AND_LONG_YTG = 7
SECOND_AND_LONG_YTG = 8
FIRST_AND_TEN_YTG = 10
SHORT_YTG = 2
SPIKE_CUTOFF_S = 60

# Field context
RED_ZONE_YARD = 20
GOAL_TO_GO_YARD = 10
MAX_YARDLINE = 100
MIN_YARDLINE = 0
MAX_DOWN = 4

# Clock heuristics (prototype; rules FSM should own final logic)
RUN_PLAY_SECONDS = 38
PASS_PLAY_SECONDS = 28
