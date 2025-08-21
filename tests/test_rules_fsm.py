from nflsim.rules.fsm import PLAY_TYPES, RulesFSM
from nflsim.state import GameState


def test_4th_down_allows_specials():
    s = GameState(2024, 2, 300, "A", "B", 4, 3, 60, 3, 3, 0, 0, 0)
    masks = RulesFSM().legal_actions(s)
    allowed = {PLAY_TYPES[i] for i, ok in enumerate(masks["play_type"]) if ok}
    assert {"punt", "fg"} <= allowed


def test_clock_never_negative():
    s = GameState(2024, 2, 5, "A", "B", 1, 10, 50, 3, 3, 0, 0, 0)
    ns = RulesFSM().apply_outcome(s, yards=0, turnover=False, score_delta=0, clock_off=5)
    assert ns.clock_bin == 0
