import argparse
import json
import os
import re
import gc
import time
from typing import List

try:
    from .eval import eval_file
    from .utils import load_model_and_tokenizer
except ImportError:
    from eval import eval_file
    from utils import load_model_and_tokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import torch
except Exception:
    torch = None


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


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
    ap.add_argument("--fewshot_strategy", type=str, default="diverse", choices=["random", "diverse"])
    ap.add_argument("--include_task_spec", type=int, default=1)
    ap.add_argument("--show_progress", type=int, default=1)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument(
        "--raw_out_prefix",
        type=str,
        default="",
        help="If set, write per-job raw predictions to files prefixed by this path.",
    )
    args = ap.parse_args()

    models = parse_csv(args.models)
    tests = parse_csv(args.tests)
    shot_values = [int(x) for x in parse_csv(args.num_shots_list)]
    show_progress = bool(args.show_progress)
    verbose = bool(args.verbose)

    rows = []
    jobs = [(m, t, s) for m in models for t in tests for s in shot_values]
    total_jobs = len(jobs)
    job_iter = jobs
    if show_progress and tqdm is not None:
        job_iter = tqdm(jobs, desc="milestone eval", unit="job", dynamic_ncols=True)

    current_model_name = None
    current_model = None
    current_tok = None

    with open(args.out, "w", encoding="utf-8") as out_f:
        for job_idx, (model, test_path, num_shots) in enumerate(job_iter, start=1):
            t_job_start = time.time()
            if verbose:
                print(
                    f"[job {job_idx}/{total_jobs} start] model={model} test={test_path} shots={num_shots}",
                    flush=True,
                )
            if model != current_model_name:
                if current_model is not None:
                    del current_model
                    del current_tok
                    current_model = None
                    current_tok = None
                    gc.collect()
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"[info] loading model: {model}", flush=True)
                current_model, current_tok = load_model_and_tokenizer(model, inference=True)
                current_model.eval()
                current_model_name = model

            raw_out_path = ""
            if args.raw_out_prefix:
                raw_out_path = (
                    f"{args.raw_out_prefix}"
                    f"_{_slugify(model)}_{_slugify(test_path)}_shot{int(num_shots)}.jsonl"
                )
                raw_dir = os.path.dirname(raw_out_path)
                if raw_dir:
                    os.makedirs(raw_dir, exist_ok=True)

            res = {
                "model": model,
                "test": test_path,
                "constrained": int(bool(args.constrained)),
                "temperature": float(args.temperature),
                "max_new_tokens": int(args.max_new_tokens),
                "num_shots": int(num_shots),
                "fewshot_strategy": str(args.fewshot_strategy),
                "include_task_spec": int(bool(args.include_task_spec)),
            }
            if raw_out_path:
                res["raw_out"] = raw_out_path
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
                    fewshot_strategy=args.fewshot_strategy,
                    include_task_spec=bool(args.include_task_spec),
                    show_progress=show_progress,
                    raw_out_path=raw_out_path,
                    model=current_model,
                    tok=current_tok,
                )
                res.update(metrics)
                res["status"] = "ok"
            except Exception as e:
                res["status"] = "error"
                res["error"] = str(e)
            rows.append(res)
            row = json.dumps(res, ensure_ascii=False)
            out_f.write(row + "\n")
            out_f.flush()
            print(row, flush=True)
            if verbose:
                elapsed = time.time() - t_job_start
                print(
                    f"[job {job_idx}/{total_jobs} done] status={res.get('status')} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if current_model is not None:
        del current_model
        del current_tok
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
