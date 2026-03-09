import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


FULL_FIELDS = [
    "source_file",
    "line_id",
    "example_idx",
    "model",
    "test_path",
    "num_shots",
    "fewshot_path",
    "fewshot_seed",
    "fewshot_strategy",
    "include_task_spec",
    "constrained",
    "max_new_tokens",
    "temperature",
    "template_id",
    "length",
    "instruction",
    "prompt",
    "raw_completion",
    "gold_program",
    "pred_program",
    "strict_valid",
    "parseable",
]

MIN_FIELDS = [
    "model",
    "num_shots",
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
                        "example_idx": obj.get("example_idx"),
                        "model": _as_str(obj.get("model")),
                        "test_path": _as_str(obj.get("test_path")),
                        "num_shots": obj.get("num_shots"),
                        "fewshot_path": _as_str(obj.get("fewshot_path")),
                        "fewshot_seed": obj.get("fewshot_seed"),
                        "fewshot_strategy": _as_str(obj.get("fewshot_strategy")),
                        "include_task_spec": obj.get("include_task_spec"),
                        "constrained": obj.get("constrained"),
                        "max_new_tokens": obj.get("max_new_tokens"),
                        "temperature": obj.get("temperature"),
                        "template_id": _as_str(obj.get("template_id")),
                        "length": obj.get("length"),
                        "instruction": _as_str(obj.get("instruction")),
                        "prompt": _as_str(obj.get("prompt")),
                        "raw_completion": _as_str(obj.get("raw_completion")),
                        "gold_program": _as_str(obj.get("gold_program")),
                        "pred_program": _as_str(obj.get("pred_program")),
                        "strict_valid": obj.get("strict_valid"),
                        "parseable": obj.get("parseable"),
                    }
                )
    return rows


def write_csv(rows: List[Dict], out_csv: Path, fields: List[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_txt(rows: List[Dict], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Total examples: {len(rows)}\n")
        f.write("=" * 80 + "\n")
        for k, r in enumerate(rows, start=1):
            f.write(f"Example {k}\n")
            f.write(f"source_file: {r['source_file']}\n")
            f.write(f"line_id: {r['line_id']}\n")
            f.write(f"example_idx: {r['example_idx']}\n")
            f.write(f"model: {r['model']}\n")
            f.write(f"test_path: {r['test_path']}\n")
            f.write(f"num_shots: {r['num_shots']}\n")
            f.write(f"fewshot_path: {r['fewshot_path']}\n")
            f.write(f"fewshot_seed: {r['fewshot_seed']}\n")
            f.write(f"fewshot_strategy: {r['fewshot_strategy']}\n")
            f.write(f"include_task_spec: {r['include_task_spec']}\n")
            f.write(f"constrained: {r['constrained']}\n")
            f.write(f"max_new_tokens: {r['max_new_tokens']}\n")
            f.write(f"temperature: {r['temperature']}\n")
            f.write(f"template_id: {r['template_id']}\n")
            f.write(f"length: {r['length']}\n")
            f.write(f"instruction: {r['instruction']}\n")
            f.write(f"prompt: {r['prompt']}\n")
            f.write(f"raw_completion: {r['raw_completion']}\n")
            f.write(f"gold_program: {r['gold_program']}\n")
            f.write(f"pred_program: {r['pred_program']}\n")
            f.write(f"strict_valid: {r['strict_valid']}\n")
            f.write(f"parseable: {r['parseable']}\n")
            f.write("-" * 80 + "\n")


def write_jsonl(rows: List[Dict], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_glob",
        type=str,
        default="outputs/*raw*.jsonl",
        help="Glob for raw prediction JSONL files.",
    )
    ap.add_argument(
        "--field_set",
        type=str,
        choices=["full", "minimal"],
        default="full",
        help="Choose full metadata CSV or minimal milestone CSV.",
    )
    ap.add_argument("--out_csv", type=str, default="outputs/human_readable.csv")
    ap.add_argument("--out_txt", type=str, default="outputs/human_readable.txt")
    ap.add_argument("--out_jsonl", type=str, default="")
    args = ap.parse_args()

    rows = load_rows(args.in_glob)
    fields = FULL_FIELDS if args.field_set == "full" else MIN_FIELDS
    write_csv(rows, Path(args.out_csv), fields=fields)
    write_txt(rows, Path(args.out_txt))
    if args.out_jsonl:
        write_jsonl(rows, Path(args.out_jsonl))
        print(f"Wrote {len(rows)} rows to {args.out_csv}, {args.out_txt}, and {args.out_jsonl}")
    else:
        print(f"Wrote {len(rows)} rows to {args.out_csv} and {args.out_txt}")


if __name__ == "__main__":
    main()
