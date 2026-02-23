import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


FIELDS = [
    "source_file",
    "line_id",
    "model",
    "test_path",
    "num_shots",
    "include_task_spec",
    "instruction",
    "gold_program",
    "pred_program",
    "strict_valid",
]


def _as_str(x) -> str:
    return "" if x is None else str(x)


def load_rows(input_glob: str) -> List[Dict]:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    rows: List[Dict] = []
    for p in paths:
        path = Path(p)
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                rows.append(
                    {
                        "source_file": str(path),
                        "line_id": i,
                        "model": _as_str(obj.get("model")),
                        "test_path": _as_str(obj.get("test_path")),
                        "num_shots": obj.get("num_shots"),
                        "include_task_spec": obj.get("include_task_spec"),
                        "instruction": _as_str(obj.get("instruction")),
                        "gold_program": _as_str(obj.get("gold_program")),
                        "pred_program": _as_str(obj.get("pred_program")),
                        "strict_valid": obj.get("strict_valid"),
                    }
                )
    return rows


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_txt(rows: List[Dict], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Total examples: {len(rows)}\n")
        f.write("=" * 80 + "\n")
        for k, r in enumerate(rows, start=1):
            f.write(f"Example {k}\n")
            f.write(f"source_file: {r['source_file']}\n")
            f.write(f"line_id: {r['line_id']}\n")
            f.write(f"model: {r['model']}\n")
            f.write(f"test_path: {r['test_path']}\n")
            f.write(f"num_shots: {r['num_shots']}\n")
            f.write(f"include_task_spec: {r['include_task_spec']}\n")
            f.write(f"instruction: {r['instruction']}\n")
            f.write(f"gold_program: {r['gold_program']}\n")
            f.write(f"pred_program: {r['pred_program']}\n")
            f.write(f"strict_valid: {r['strict_valid']}\n")
            f.write("-" * 80 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_glob",
        type=str,
        default="outputs/*raw*.jsonl",
        help="Glob for raw prediction JSONL files.",
    )
    ap.add_argument("--out_csv", type=str, default="outputs/human_readable.csv")
    ap.add_argument("--out_txt", type=str, default="outputs/human_readable.txt")
    args = ap.parse_args()

    rows = load_rows(args.in_glob)
    write_csv(rows, Path(args.out_csv))
    write_txt(rows, Path(args.out_txt))
    print(f"Wrote {len(rows)} rows to {args.out_csv} and {args.out_txt}")


if __name__ == "__main__":
    main()
