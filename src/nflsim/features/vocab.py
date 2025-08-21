"""
Centralized enums/bins so constants aren't duplicated.
Import from here instead of redefining in multiple modules.
"""

PLAY_TYPES = ("run", "pass", "punt", "fg", "spike", "kneel", "ko", "onside", "penalty")

YARDLINE_TOKENS = list(range(0, 101))  # 0..100
CLOCK_BINS_5S = list(range(0, 720))  # 0..719 (5s bins per quarter)
DISTANCE_BUCKETS = list(range(1, 21)) + ["20+"]  # 1..20 + 20+
SCORE_DIFF_2PT = list(range(-14, 15, 2))  # -14..+14 in 2-pt steps
