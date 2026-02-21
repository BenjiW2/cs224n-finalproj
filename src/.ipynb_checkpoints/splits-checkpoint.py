# src/splits.py
import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

from .actions import contains_forbidden_pair

# 8 forbidden ordered adjacent tool pairs
FORBIDDEN: Set[Tuple[str, str]] = {
    ("left", "forward"),
    ("right", "forward"),
    ("forward", "left"),
    ("forward", "right"),
    ("backward", "left"),
    ("backward", "right"),
    ("forward", "bark"),
    ("left", "bark"),
}

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stats(rows: List[Dict], name: str):
    lens = Counter(r["length"] for r in rows)
    tools = Counter()
    forb = 0
    for r in rows:
        acts = [(a[0], a[1]) for a in r["actions"]]
        for t,_ in acts:
            tools[t] += 1
        if contains_forbidden_pair(acts, FORBIDDEN):
            forb += 1
    print(f"\n[{name}] n={len(rows)}")
    print(" length:", dict(sorted(lens.items())))
    print(" tools:", dict(tools))
    print(" forbidden-containing:", forb)

def match_length_distribution(source: List[Dict], target_len_counts: Counter, rng: random.Random) -> List[Dict]:
    by_len = defaultdict(list)
    for r in source:
        by_len[r["length"]].append(r)
    out = []
    for L, cnt in target_len_counts.items():
        pool = by_len[L]
        rng.shuffle(pool)
        out.extend(pool[:cnt])
    rng.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iid_train", type=int, default=100000)
    ap.add_argument("--iid_dev", type=int, default=10000)
    ap.add_argument("--iid_test", type=int, default=10000)
    ap.add_argument("--template_holdout", action="store_true")
    ap.add_argument("--holdout_templates", type=str, default="imperative")  # comma-separated
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.infile)
    rng.shuffle(rows)

    # Optional template holdout: train on templates != holdout; test on holdout templates
    holdout_templates = set([t.strip() for t in args.holdout_templates.split(",") if t.strip()])
    if args.template_holdout:
        train_pool = [r for r in rows if r["template_id"] not in holdout_templates]
        test_pool_tpl = [r for r in rows if r["template_id"] in holdout_templates]
    else:
        train_pool = rows
        test_pool_tpl = []

    # IID split
    iid_train = train_pool[:args.iid_train]
    iid_dev = train_pool[args.iid_train:args.iid_train + args.iid_dev]
    iid_test = train_pool[args.iid_train + args.iid_dev: args.iid_train + args.iid_dev + args.iid_test]

    write_jsonl(f"{args.outdir}/iid_train.jsonl", iid_train)
    write_jsonl(f"{args.outdir}/iid_dev.jsonl", iid_dev)
    write_jsonl(f"{args.outdir}/iid_test.jsonl", iid_test)
    stats(iid_train, "iid_train")
    stats(iid_test, "iid_test")

    # Length generalization: train length=1; test length 2-4
    len_train = [r for r in iid_train if r["length"] == 1]
    len_test = [r for r in iid_test if r["length"] >= 2]
    write_jsonl(f"{args.outdir}/len_train.jsonl", len_train)
    write_jsonl(f"{args.outdir}/len_test.jsonl", len_test)
    stats(len_train, "len_train")
    stats(len_test, "len_test")

    # Held-out composition:
    held_train = []
    held_test = []
    held_control_pool = []

    for r in iid_train:
        acts = [(a[0], a[1]) for a in r["actions"]]
        if not contains_forbidden_pair(acts, FORBIDDEN):
            held_train.append(r)

    for r in iid_test:
        acts = [(a[0], a[1]) for a in r["actions"]]
        if contains_forbidden_pair(acts, FORBIDDEN):
            held_test.append(r)
        else:
            held_control_pool.append(r)

    # Control: match length distribution to held_test
    target_len = Counter(r["length"] for r in held_test)
    held_control = match_length_distribution(held_control_pool, target_len, rng)

    write_jsonl(f"{args.outdir}/held_train.jsonl", held_train)
    write_jsonl(f"{args.outdir}/held_test.jsonl", held_test)
    write_jsonl(f"{args.outdir}/held_control.jsonl", held_control)
    stats(held_train, "held_train")
    stats(held_test, "held_test")
    stats(held_control, "held_control")

    # Template holdout test set if requested
    if args.template_holdout:
        # use only length>=2 as the challenge, optional
        write_jsonl(f"{args.outdir}/template_holdout_test.jsonl", test_pool_tpl[:args.iid_test])
        stats(test_pool_tpl[:args.iid_test], "template_holdout_test")

if __name__ == "__main__":
    main()