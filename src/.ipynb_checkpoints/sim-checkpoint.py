# src/sim.py
import math
from typing import List, Tuple

# Bin mappings (shared bins for move + turn)
DIST_MAP = {10: 1.0, 30: 3.0, 60: 6.0, 100: 10.0}
TURN_MAP = {10: 15.0, 30: 45.0, 60: 90.0, 100: 180.0}

State = Tuple[float, float, float]  # x, y, heading_deg
Action = Tuple[str, int]            # tool, val

def _heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg % 360.0)
    return (math.cos(rad), math.sin(rad))

def execute(actions: List[Action], start: State = (0.0, 0.0, 0.0)) -> List[State]:
    """Execute actions and return trajectory of states after each step."""
    x, y, h = start
    traj: List[State] = []
    for tool, val in actions:
        if tool == "forward":
            d = DIST_MAP[val]
            ux, uy = _heading_to_unit(h)
            x += d * ux
            y += d * uy
        elif tool == "backward":
            d = DIST_MAP[val]
            ux, uy = _heading_to_unit(h)
            x -= d * ux
            y -= d * uy
        elif tool == "left":
            h = (h + TURN_MAP[val]) % 360.0
        elif tool == "right":
            h = (h - TURN_MAP[val]) % 360.0
        elif tool == "bark":
            # no-op on state
            pass
        else:
            raise ValueError(f"Unknown tool {tool}")
        traj.append((x, y, h))
    return traj

def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return abs(d)

def trajectory_score(traj: List[State], ref: List[State], alpha: float = 0.35, beta: float = 0.03) -> float:
    """
    Score in [0,1]. Compares step-by-step position + heading.
    exp(-alpha*pos_err) * exp(-beta*heading_err)
    """
    if len(traj) == 0 and len(ref) == 0:
        return 1.0
    # If lengths differ, pad comparison to min length and penalize length mismatch.
    L = min(len(traj), len(ref))
    if L == 0:
        return 0.0
    scores = []
    for i in range(L):
        x,y,h = traj[i]
        xr,yr,hr = ref[i]
        pos_err = math.sqrt((x-xr)**2 + (y-yr)**2)
        head_err = _angle_diff_deg(h, hr)
        s = math.exp(-alpha*pos_err) * math.exp(-beta*head_err)
        scores.append(s)
    base = sum(scores) / len(scores)
    # length penalty (mild): exact length gets 1.0 multiplier
    len_pen = math.exp(-0.6 * abs(len(traj) - len(ref)))
    return float(base * len_pen)

if __name__ == "__main__":
    # quick sanity
    ref = execute([("left",30),("forward",60),("bark",0)])
    cand = execute([("left",30),("forward",60),("bark",0)])
    print("score same:", trajectory_score(cand, ref))
    cand2 = execute([("right",30),("forward",60),("bark",0)])
    print("score diff:", trajectory_score(cand2, ref))
    print("sim.py ok")