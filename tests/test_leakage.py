import pandas as pd
from nflsim.data.schema import enforce_whitelist, LEAKY_COLUMNS

def test_leakage_guard_drops_columns():
    df = pd.DataFrame({
        'season':[2024],'wp':[0.5],'yardline_100':[50],'down':[1],'ydstogo':[10],'qtr':[1],
        'half_seconds_remaining':[1800],'game_seconds_remaining':[3600],'score_differential':[0],
        'posteam_timeouts_remaining':[3],'defteam_timeouts_remaining':[3],'personnel_offense':['11'],
        'personnel_defense':['4-2-5'],'posteam':['KC'],'defteam':['SF'],'home_team':['KC'],'away_team':['SF'],
        'shotgun':[1],'no_huddle':[0],'play_id':[1],'game_id':['g1'],'play_type':['run'],
        'pass_length':[None],'run_location':['middle'],'yards_gained':[3],'timeout':[0],'td_team':[None],
        'field_goal_result':[None]
    })
    out = enforce_whitelist(df)
    assert all(col not in out.columns for col in LEAKY_COLUMNS)
