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

def row_key(row: Dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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
    shortages = {}
    for L, cnt in target_len_counts.items():
        pool = list(by_len[L])
        rng.shuffle(pool)
        if len(pool) < cnt:
            shortages[L] = (len(pool), cnt)
        out.extend(pool[:cnt])
    if shortages:
        detail = ", ".join(
            f"len={length}: available={available}, needed={needed}"
            for length, (available, needed) in sorted(shortages.items())
        )
        raise ValueError(
            "Insufficient source rows to match target length distribution for held_control "
            f"({detail}). Increase the source data or reduce the requested IID split sizes."
        )
    rng.shuffle(out)
    return out

def non_forbidden_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        acts = [(a[0], a[1]) for a in r["actions"]]
        if not contains_forbidden_pair(acts, FORBIDDEN):
            out.append(r)
    return out

def subtract_used_rows(rows: List[Dict], used_rows: List[Dict]) -> List[Dict]:
    used_counts = Counter(row_key(r) for r in used_rows)
    out = []
    for r in rows:
        key = row_key(r)
        if used_counts[key] > 0:
            used_counts[key] -= 1
        else:
            out.append(r)
    return out

def rebuild_held_control(
    all_rows: List[Dict],
    iid_train: List[Dict],
    iid_dev: List[Dict],
    iid_test: List[Dict],
    held_test: List[Dict],
    rng: random.Random,
) -> List[Dict]:
    iid_test_pool = non_forbidden_rows(iid_test)
    unused_rows = subtract_used_rows(all_rows, iid_train + iid_dev + iid_test)
    fallback_pool = non_forbidden_rows(unused_rows)
    held_control_pool = iid_test_pool + fallback_pool
    target_len = Counter(r["length"] for r in held_test)
    return match_length_distribution(held_control_pool, target_len, rng)

def rebuild_held_control_only(args, rng: random.Random):
    all_rows = load_jsonl(args.infile)
    held_test_path = args.held_test_path or f"{args.outdir}/held_test.jsonl"
    iid_train_path = args.iid_train_path or f"{args.outdir}/iid_train.jsonl"
    iid_dev_path = args.iid_dev_path or f"{args.outdir}/iid_dev.jsonl"
    iid_test_path = args.iid_test_path or f"{args.outdir}/iid_test.jsonl"
    held_control_path = args.held_control_path or f"{args.outdir}/held_control.jsonl"

    iid_train = load_jsonl(iid_train_path)
    iid_dev = load_jsonl(iid_dev_path)
    iid_test = load_jsonl(iid_test_path)
    held_test = load_jsonl(held_test_path)

    held_control = rebuild_held_control(all_rows, iid_train, iid_dev, iid_test, held_test, rng)
    write_jsonl(held_control_path, held_control)
    stats(held_test, "held_test_existing")
    stats(held_control, "held_control_rebuilt")

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
    ap.add_argument("--held_control_only", action="store_true")
    ap.add_argument("--iid_train_path", type=str, default=None)
    ap.add_argument("--iid_dev_path", type=str, default=None)
    ap.add_argument("--iid_test_path", type=str, default=None)
    ap.add_argument("--held_test_path", type=str, default=None)
    ap.add_argument("--held_control_path", type=str, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.held_control_only:
        rebuild_held_control_only(args, rng)
        return

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
    held_train = non_forbidden_rows(iid_train)
    held_test = [r for r in iid_test if contains_forbidden_pair([(a[0], a[1]) for a in r["actions"]], FORBIDDEN)]
    held_control = rebuild_held_control(rows, iid_train, iid_dev, iid_test, held_test, rng)

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
