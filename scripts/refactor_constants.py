from __future__ import annotations
from pathlib import Path
import re

FILES = [
    Path("src/nflsim/decoding/simulate.py"),
    Path("src/nflsim/training/train_tiny.py"),
]

CONST_IMPORT = (
    "from nflsim.constants import (THIRD_AND_LONG_YTG, SECOND_AND_LONG_YTG, "
    "FIRST_AND_TEN_YTG, SHORT_YTG, SPIKE_CUTOFF_S, RED_ZONE_YARD, GOAL_TO_GO_YARD, "
    "RUN_PLAY_SECONDS, PASS_PLAY_SECONDS)\n"
)

def ensure_const_import(txt: str) -> str:
    if "from nflsim.constants import" in txt:
        return txt
    lines = txt.splitlines(keepends=True)
    # insert after imports (after __future__ and any import/from lines)
    j = 0
    for i, ln in enumerate(lines[:40]):
        if ln.startswith("from __future__"):
            j = i + 1
    while j < len(lines) and (lines[j].lstrip().startswith("import ") or lines[j].lstrip().startswith("from ")):
        j += 1
    lines.insert(j, CONST_IMPORT)
    return "".join(lines)

def split_semicolons(txt: str) -> str:
    out = []
    for ln in txt.splitlines():
        if ";" in ln and not ln.strip().startswith("#"):
            indent = re.match(r"^(\s*)", ln).group(1)
            parts = [p.strip() for p in ln.split(";") if p.strip()]
            for p in parts:
                out.append(f"{indent}{p}")
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if not txt.endswith("\n") else "")

def expand_inline_if(txt: str) -> str:
    # turn: if cond: stmt   -> two lines
    pat = re.compile(r"^(\s*)(if|elif|else)\s*(.*?):\s+(.+)$")
    out = []
    for ln in txt.splitlines():
        m = pat.match(ln)
        if m and m.group(2) in ("if","elif"):
            indent, kw, cond, stmt = m.groups()
            out += [f"{indent}{kw} {cond}:", f"{indent}    {stmt}"]
        elif m and m.group(2) == "else":
            indent, _, _, stmt = m.groups()
            out += [f"{indent}else:", f"{indent}    {stmt}"]
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if not txt.endswith("\n") else "")

def replace_magic_numbers(txt: str) -> str:
    txt = re.sub(r"\bydstogo\s*>=\s*8\b", "ydstogo >= SECOND_AND_LONG_YTG", txt)
    txt = re.sub(r"\bydstogo\s*==\s*10\b", "ydstogo == FIRST_AND_TEN_YTG", txt)
    txt = re.sub(r"\bydstogo\s*<=\s*2\b", "ydstogo <= SHORT_YTG", txt)
    txt = re.sub(r"\bgame_seconds_remaining\s*>\s*60\b", "game_seconds_remaining > SPIKE_CUTOFF_S", txt)
    txt = re.sub(r"\byardline_100\s*<=\s*10\b", "yardline_100 <= GOAL_TO_GO_YARD", txt)
    txt = re.sub(r"\byardline_100\s*<=\s*20\b", "yardline_100 <= RED_ZONE_YARD", txt)
    # clock deltas in simulate.py
    txt = re.sub(
        r"game_seconds_remaining\s*-\s*\(\s*38\s*if\s+play_type\s+in\s*\(\s*\"run\",\s*\"kneel\"\s*\)\s*else\s*28\s*\)",
        "game_seconds_remaining - (RUN_PLAY_SECONDS if play_type in (\"run\",\"kneel\") else PASS_PLAY_SECONDS)",
        txt,
    )
    return txt

def clean_yards_casts(txt: str) -> str:
    return txt.replace("int(round(yards))", "yards")

for path in FILES:
    if not path.exists():
        continue
    s = path.read_text()
    s = ensure_const_import(s)
    s = expand_inline_if(s)
    s = split_semicolons(s)
    s = replace_magic_numbers(s)
    s = clean_yards_casts(s)
    path.write_text(s)

# tests: fix Yoda-style asserts if present
p = Path("tests/test_small_e2e.py")
if p.exists():
    ts = p.read_text()
    ts = ts.replace("assert 0 <= s.clock_bin", "assert s.clock_bin >= 0")
    ts = ts.replace("assert 0 <= s.yardline <= 100", "assert s.yardline >= 0 and s.yardline <= 100")
    p.write_text(ts)
