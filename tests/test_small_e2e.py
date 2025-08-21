import numpy as np
from nflsim.state import GameState
from nflsim.rules.fsm import RulesFSM

def test_e2e_random_policy_invariants():
    rng = np.random.default_rng(0)
    fsm = RulesFSM()
    s = GameState(2024,1,700,"A","B",1,10,25,3,3,0,0,0)
    for _ in range(100):
        masks = fsm.legal_actions(s)
        assert masks["play_type"].any()
        s = fsm.apply_outcome(s, yards=int(rng.integers(-3,8)), turnover=False, score_delta=0, clock_off=5)
        assert 0 <= s.yardline <= 100
        assert 0 <= s.clock_bin
