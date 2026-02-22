import argparse
import json
from collections import Counter
from typing import Dict, List, Tuple

try:
    from .actions import is_valid, parse, parse_loose, first_invalid_reason
    from .sim import execute, trajectory_score
except ImportError:
    from actions import is_valid, parse, parse_loose, first_invalid_reason
    from sim import execute, trajectory_score


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_predictions(path: str, pred_key: str) -> List[str]:
    preds: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                preds.append("")
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                preds.append(raw.strip())
                continue

            if isinstance(item, str):
                preds.append(item.strip())
            elif isinstance(item, dict):
                if pred_key in item:
                    preds.append(str(item[pred_key]).strip())
                elif "pred_program" in item:
                    preds.append(str(item["pred_program"]).strip())
                elif "program" in item:
                    preds.append(str(item["program"]).strip())
                else:
                    preds.append("")
            else:
                preds.append("")
    return preds


def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def step_f1(pred: List[Tuple[str, int]], gold: List[Tuple[str, int]]) -> Tuple[float, float, float]:
    L = max(len(pred), len(gold))
    if L == 0:
        return 1.0, 1.0, 1.0
    tp = 0
    for i in range(min(len(pred), len(gold))):
        if pred[i] == gold[i]:
            tp += 1
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(gold), 1)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return prec, rec, f1


def score(gold_rows: List[Dict], preds: List[str]) -> Dict:
    if len(preds) < len(gold_rows):
        preds = preds + [""] * (len(gold_rows) - len(preds))
    elif len(preds) > len(gold_rows):
        preds = preds[: len(gold_rows)]

    metrics = Counter()
    invalid_reasons = Counter()
    precs, recs, f1s = [], [], []
    edit_tools = []
    length_ok = 0
    traj_scores = []

    for row, pred_prog in zip(gold_rows, preds):
        gold_prog = str(row["program"]).strip()
        gold_actions = parse(gold_prog)

        metrics["total"] += 1

        if is_valid(pred_prog):
            metrics["valid"] += 1
        else:
            invalid_reasons[first_invalid_reason(pred_prog)] += 1

        pred_actions = parse_loose(pred_prog)
        if pred_actions is not None:
            metrics["parseable"] += 1

        if pred_actions is not None and gold_actions is not None:
            if pred_prog.strip() == gold_prog:
                metrics["exact"] += 1

            p, r, f1 = step_f1(pred_actions, gold_actions)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

            pred_tools = [t for (t, _) in pred_actions]
            gold_tools = [t for (t, _) in gold_actions]
            edit_tools.append(levenshtein(pred_tools, gold_tools))

            if len(pred_actions) == len(gold_actions):
                length_ok += 1

            traj = execute(pred_actions)
            traj_ref = execute(gold_actions)
            traj_scores.append(trajectory_score(traj, traj_ref))

    total = metrics["total"]
    parsed = metrics["parseable"]
    out = {
        "total": total,
        "valid_rate": metrics["valid"] / max(total, 1),
        "parseable_rate": parsed / max(total, 1),
        "exact_match": metrics["exact"] / max(total, 1),
        "step_precision": sum(precs) / max(len(precs), 1),
        "step_recall": sum(recs) / max(len(recs), 1),
        "step_f1": sum(f1s) / max(len(f1s), 1),
        "tool_edit_dist": sum(edit_tools) / max(len(edit_tools), 1),
        "length_acc": length_ok / max(total, 1),
        "mean_traj_score": sum(traj_scores) / max(len(traj_scores), 1),
        "invalid_reasons": dict(invalid_reasons),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, required=True, help="Gold JSONL with `program` field.")
    ap.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Predictions file: JSONL rows with `pred_key` or raw programs, one per line.",
    )
    ap.add_argument("--pred_key", type=str, default="pred_program")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    preds = read_predictions(args.pred, pred_key=args.pred_key)
    res = score(gold_rows, preds)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
