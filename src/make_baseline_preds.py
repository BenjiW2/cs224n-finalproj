import argparse
import json
import random
from typing import Dict, List, Tuple

try:
    from .actions import MOVE_TOOLS, EVENT_TOOLS, BINS, parse, serialize
except ImportError:
    from actions import MOVE_TOOLS, EVENT_TOOLS, BINS, parse, serialize


Action = Tuple[str, int]


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_program(rng: random.Random, length: int) -> List[Action]:
    actions: List[Action] = []
    for _ in range(length):
        tool = rng.choice(MOVE_TOOLS + EVENT_TOOLS)
        val = rng.choice(BINS) if tool in MOVE_TOOLS else 0
        actions.append((tool, val))
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["oracle", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--match_gold_length",
        type=int,
        default=1,
        help="For random mode, sample sequence lengths to match each gold example.",
    )
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold_rows = read_jsonl(args.gold)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in gold_rows:
            gold_prog = str(row["program"]).strip()

            if args.mode == "oracle":
                pred_prog = gold_prog
            else:
                gold_actions = parse(gold_prog)
                if bool(args.match_gold_length) and gold_actions is not None:
                    length = len(gold_actions)
                else:
                    length = rng.randint(args.min_len, args.max_len)
                pred_prog = serialize(sample_program(rng, length))

            out_row = {
                "instruction": row.get("instruction", ""),
                "pred_program": pred_prog,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(gold_rows)} predictions to {args.out} ({args.mode})")


if __name__ == "__main__":
    main()
