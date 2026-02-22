import argparse
import json
from typing import List

try:
    from .eval import eval_file
except ImportError:
    from eval import eval_file


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, required=True, help="Comma-separated model names/paths.")
    ap.add_argument("--tests", type=str, required=True, help="Comma-separated test JSONL files.")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path.")

    ap.add_argument("--constrained", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--fewshot_path", type=str, default="")
    ap.add_argument("--num_shots_list", type=str, default="0")
    ap.add_argument("--fewshot_seed", type=int, default=0)
    args = ap.parse_args()

    models = parse_csv(args.models)
    tests = parse_csv(args.tests)
    shot_values = [int(x) for x in parse_csv(args.num_shots_list)]

    rows = []
    for model in models:
        for test_path in tests:
            for num_shots in shot_values:
                res = {
                    "model": model,
                    "test": test_path,
                    "constrained": int(bool(args.constrained)),
                    "temperature": float(args.temperature),
                    "max_new_tokens": int(args.max_new_tokens),
                    "num_shots": int(num_shots),
                }
                try:
                    metrics = eval_file(
                        model_path=model,
                        test_path=test_path,
                        constrained=bool(args.constrained),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        fewshot_path=args.fewshot_path,
                        num_shots=num_shots,
                        fewshot_seed=args.fewshot_seed,
                    )
                    res.update(metrics)
                    res["status"] = "ok"
                except Exception as e:
                    res["status"] = "error"
                    res["error"] = str(e)
                rows.append(res)
                print(json.dumps(res, ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
