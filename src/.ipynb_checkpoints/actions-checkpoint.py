# src/actions.py
import re
from typing import List, Optional, Tuple, Dict, Set

MOVE_TOOLS = ["forward", "backward", "left", "right"]
EVENT_TOOLS = ["bark"]  # keep v1 simple; add "stop" later if desired
ALL_TOOLS = MOVE_TOOLS + EVENT_TOOLS

BINS = [10, 30, 60, 100]
BARK_VAL = 0

# Strict regex: concatenation of valid calls only.
PROGRAM_RE = re.compile(
    r"^(\[(forward|backward|left|right):(10|30|60|100)\]|\[bark:0\])+$"
)

def is_valid(program: str) -> bool:
    if program is None:
        return False
    s = program.strip()
    return bool(PROGRAM_RE.match(s))

def parse(program: str) -> Optional[List[Tuple[str, int]]]:
    """Parse bracket program -> list of (tool, value) or None if invalid."""
    if not is_valid(program):
        return None
    s = program.strip()
    out: List[Tuple[str, int]] = []
    i = 0
    while i < len(s):
        if s[i] != "[":
            return None
        j = s.find("]", i)
        if j == -1:
            return None
        chunk = s[i+1:j]  # e.g. left:30
        if ":" not in chunk:
            return None
        tool, val_s = chunk.split(":", 1)
        try:
            val = int(val_s)
        except ValueError:
            return None
        # enforce tool/val constraints
        if tool in MOVE_TOOLS:
            if val not in BINS:
                return None
        elif tool in EVENT_TOOLS:
            if tool == "bark" and val != 0:
                return None
        else:
            return None
        out.append((tool, val))
        i = j + 1
    return out

def serialize(actions: List[Tuple[str, int]]) -> str:
    """List of (tool,val) -> canonical bracket string."""
    parts = []
    for tool, val in actions:
        parts.append(f"[{tool}:{val}]")
    return "".join(parts)

def adjacent_pairs(actions: List[Tuple[str, int]]) -> List[Tuple[str, str]]:
    """Adjacent tool bigrams (tool_i, tool_{i+1})."""
    tools = [t for (t, _) in actions]
    return list(zip(tools, tools[1:]))

def contains_forbidden_pair(actions: List[Tuple[str, int]], forbidden: Set[Tuple[str, str]]) -> bool:
    return any(p in forbidden for p in adjacent_pairs(actions))

def first_invalid_reason(program: str) -> str:
    """Useful for debugging format failures."""
    if program is None:
        return "none"
    s = program.strip()
    if not s.startswith("["):
        return "no_open_bracket"
    if not s.endswith("]"):
        return "no_close_bracket"
    if not PROGRAM_RE.match(s):
        return "regex_mismatch"
    if parse(s) is None:
        return "parse_failed"
    return "ok"

if __name__ == "__main__":
    # Basic tests
    good = "[left:30][forward:60][bark:0]"
    assert is_valid(good)
    assert parse(good) == [("left",30),("forward",60),("bark",0)]
    bad1 = "[left:45]"
    assert not is_valid(bad1)
    bad2 = "left:30"
    assert not is_valid(bad2)
    print("actions.py ok")