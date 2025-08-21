from nflsim.rules.fsm import RulesFSM
from nflsim.state import GameState


def test_masks_never_empty_for_play_type():
    s = GameState(2024, 1, 600, "A", "B", 1, 10, 50, 3, 3, 0, 0, 0)
    masks = RulesFSM().legal_actions(s)
    assert masks["play_type"].any()
