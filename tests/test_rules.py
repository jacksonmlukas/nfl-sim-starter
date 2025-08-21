from nflsim.rules.fsm import RulesFSM
from nflsim.state import GameState


def test_legal_basic():
    s = GameState(2024, 1, 700, "A", "B", 1, 10, 25, 3, 3, 0, 0, 0)
    fsm = RulesFSM()
    masks = fsm.legal_actions(s)
    assert masks["play_type"].any()
