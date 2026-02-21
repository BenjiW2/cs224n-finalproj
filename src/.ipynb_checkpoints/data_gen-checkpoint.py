import argparse
import json
import random
from typing import List, Tuple, Dict

from .actions import MOVE_TOOLS, EVENT_TOOLS, BINS, serialize

Action = Tuple[str, int]

# Deterministic phrase mappings (value -> phrase options)
INTENSITY = {
    10: ["a tiny bit", "slightly", "just a little"],
    30: ["a bit", "a little"],
    60: ["for a while", "some distance"],
    100: ["a lot", "for a long while"],
}
MOVE_SYNS = {
    "forward": ["go forward", "move forward", "advance", "move ahead"],
    "backward": ["go backward", "move back", "back up", "reverse"],
    "left": ["turn left", "rotate left", "veer left"],
    "right": ["turn right", "rotate right", "veer right"],
}
BARK_SYNS = ["bark", "make a bark sound", "let out a bark"]

CONNECTORS = ["then", "after that", "and then", "next"]
FILLERS = ["", "", "", "please", "okay", "um", "kind of", "carefully"]

# Template functions return instruction strings; they also define a "template_id"
def template_linear(actions: List[Action], rng: random.Random) -> str:
    # e.g., "Turn left a bit, then go forward for a while, then bark."
    parts = []
    for (tool, val) in actions:
        filler = rng.choice(FILLERS).strip()
        if tool in MOVE_SYNS:
            verb = rng.choice(MOVE_SYNS[tool])
            phrase = rng.choice(INTENSITY[val])
            chunk = f"{verb} {phrase}"
        else:
            chunk = rng.choice(BARK_SYNS)
        if filler:
            chunk = f"{filler} {chunk}"
        parts.append(chunk.strip())
    out = []
    for i, p in enumerate(parts):
        if i == 0:
            out.append(p)
        else:
            out.append(f"{rng.choice(CONNECTORS)} {p}")
    s = ", ".join(out)
    # light punctuation variation
    if rng.random() < 0.5:
        s += "."
    return s

def template_imperative(actions: List[Action], rng: random.Random) -> str:
    # e.g., "First, rotate left slightly. After that, advance for a while. Next, bark."
    parts = []
    ords = ["First", "Then", "After that", "Next"]
    for idx, (tool, val) in enumerate(actions):
        if tool in MOVE_SYNS:
            verb = rng.choice(MOVE_SYNS[tool])
            phrase = rng.choice(INTENSITY[val])
            chunk = f"{verb} {phrase}"
        else:
            chunk = rng.choice(BARK_SYNS)
        prefix = ords[min(idx, len(ords)-1)]
        parts.append(f"{prefix}, {chunk}")
    return " ".join(p + "." for p in parts)

TEMPLATES = {
    "linear": template_linear,
    "imperative": template_imperative,
}

def sample_program(rng: random.Random, length: int) -> List[Action]:
    actions: List[Action] = []
    for _ in range(length):
        tool = rng.choice(MOVE_TOOLS + EVENT_TOOLS)
        if tool in MOVE_TOOLS:
            val = rng.choice(BINS)
        else:
            val = 0
        actions.append((tool, val))
    return actions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_programs", type=int, default=5000)
    ap.add_argument("--k_paraphrases", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows: List[Dict] = []
    for _ in range(args.n_programs):
        L = rng.randint(args.min_len, args.max_len)
        prog_actions = sample_program(rng, L)
        prog_str = serialize(prog_actions)
        for _k in range(args.k_paraphrases):
            template_id = rng.choice(list(TEMPLATES.keys()))
            instr = TEMPLATES[template_id](prog_actions, rng)
            rows.append({
                "instruction": instr,
                "program": prog_str,
                "actions": [[t, v] for (t, v) in prog_actions],
                "length": L,
                "template_id": template_id,
            })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {args.out}")

if __name__ == "__main__":
    main()