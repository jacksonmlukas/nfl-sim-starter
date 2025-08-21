from nflsim.rules.fsm import State, legal_play_types, clamp_state

def test_legal_basic():
    s = State(1, 3600, 1, 10, 75, 3, 3, 'home', 0, 0)
    legal = legal_play_types(s)
    assert 'run' in legal and 'pass' in legal

def test_clamp_bounds():
    s = State(1, -5, 8, 0, 150, 3, 3, 'home', 0, 0)
    s = clamp_state(s)
    assert 0 <= s.game_seconds_remaining
    assert 1 <= s.down <= 4
    assert 0 <= s.yardline_100 <= 100
