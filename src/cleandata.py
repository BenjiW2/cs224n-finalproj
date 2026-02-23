import json
import csv
from pathlib import Path

# Input: your pasted content saved as a .jsonl (one JSON object per line)
IN_PATH  = Path("outputs/qwen06b_raw_tiny_v2_all.jsonl")   # change if needed
OUT_PATH = Path("runs.csv")

rows = []
with IN_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({
            "num_shots": obj.get("num_shots"),
            "instruction": obj.get("instruction"),
            "gold_program": obj.get("gold_program"),
            "completion_raw": obj.get("pred_program"),
            "strict_valid": obj.get("strict_valid"),
        })

with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["num_shots","instruction","gold_program","completion_raw","strict_valid"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_PATH}")