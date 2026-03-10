import argparse
import csv
import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from .actions import parse_loose
except ImportError:
    from actions import parse_loose


Action = Tuple[str, int]


def read_raw_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_actions(program: str) -> List[Action]:
    actions = parse_loose(str(program or "").strip())
    return actions or []


def basename_split(test_path: str) -> str:
    name = str(test_path).strip().split("/")[-1]
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def position_metrics(rows: List[Dict], max_pos: int) -> List[Dict]:
    first = rows[0] if rows else {}
    out: List[Dict] = []
    parsed = []
    for r in rows:
        gold = parse_actions(r.get("gold_program", ""))
        pred = parse_actions(r.get("pred_program", ""))
        parsed.append((gold, pred))

    for pos in range(1, max_pos + 1):
        idx = pos - 1
        gold_present = 0
        pred_present = 0
        exact_ok = 0
        tool_ok = 0
        exact_ok_given_present = 0
        tool_ok_given_present = 0

        for gold, pred in parsed:
            if len(gold) <= idx:
                continue
            gold_present += 1
            if len(pred) > idx:
                pred_present += 1
                if pred[idx] == gold[idx]:
                    exact_ok += 1
                    exact_ok_given_present += 1
                if pred[idx][0] == gold[idx][0]:
                    tool_ok += 1
                    tool_ok_given_present += 1

        pred_present_den = max(pred_present, 1)
        gold_present_den = max(gold_present, 1)
        out.append(
            {
                "model": first.get("model", ""),
                "model_short": str(first.get("model", "")).split("/")[-1],
                "test_path": first.get("test_path", ""),
                "split": basename_split(first.get("test_path", "")),
                "num_shots": first.get("num_shots", ""),
                "raw_path": first.get("raw_path", ""),
                "position": pos,
                "gold_present": gold_present,
                "pred_present": pred_present,
                "pred_present_rate": pred_present / gold_present_den,
                "call_acc": exact_ok / gold_present_den,
                "tool_acc": tool_ok / gold_present_den,
                "call_acc_given_present": exact_ok_given_present / pred_present_den,
                "tool_acc_given_present": tool_ok_given_present / pred_present_den,
            }
        )
    return out


def aggregate(rows: List[Dict], key_fields: List[str]) -> List[Dict]:
    buckets: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        buckets[key].append(row)

    out: List[Dict] = []
    metric_fields = [
        "gold_present",
        "pred_present",
        "pred_present_rate",
        "call_acc",
        "tool_acc",
        "call_acc_given_present",
        "tool_acc_given_present",
    ]
    for key, group in sorted(buckets.items()):
        item = {k: v for k, v in zip(key_fields, key)}
        for mf in metric_fields:
            item[mf] = sum(float(r[mf]) for r in group) / len(group)
        item["n_rows"] = len(group)
        out.append(item)
    return out


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_glob", required=True)
    ap.add_argument("--max_pos", type=int, default=3)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_agg_csv", default="")
    ap.add_argument(
        "--agg_keys",
        default="model_short,num_shots,position",
        help="Comma-separated keys for aggregated CSV.",
    )
    args = ap.parse_args()

    raw_paths = sorted(glob.glob(args.raw_glob))
    if not raw_paths:
        raise FileNotFoundError(f"No raw files matched: {args.raw_glob}")

    rows: List[Dict] = []
    for raw_path in raw_paths:
        raw_rows = read_raw_rows(raw_path)
        metrics = position_metrics(raw_rows, args.max_pos)
        for m in metrics:
            m["raw_path"] = raw_path
            rows.append(m)

    if args.out_csv:
        write_csv(args.out_csv, rows)

    if args.out_agg_csv:
        agg_keys = [x.strip() for x in args.agg_keys.split(",") if x.strip()]
        agg_rows = aggregate(rows, agg_keys)
        write_csv(args.out_agg_csv, agg_rows)


if __name__ == "__main__":
    main()
