import argparse
import ast
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
LOSS_LINE_RE = re.compile(r"\{.*'(?:loss|eval_loss)'.*\}")


RUNS = [
    "qwen06b_sft_iid_18k",
    "qwen06b_sft_iid_18k_stepweighted",
    "qwen17b_sft_iid_18k",
    "qwen17b_base_sft_iid_18k",
    "qwen4b_sft_iid_18k",
]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def existing_csv_for_run(run_name: str) -> Optional[Path]:
    candidates = [
        Path(f"finalproj_outputs/outputs/{run_name}/loss_curve.csv"),
        Path(f"outputs/{run_name}/loss_curve.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def existing_log_for_run(run_name: str) -> Optional[Path]:
    candidates = [
        Path(f"finalproj_outputs/outputs/final_logs/final_logs/{run_name}.log"),
        Path(f"finalproj_outputs/outputs/final_logs/{run_name}.log"),
        Path(f"outputs/final_logs/{run_name}.log"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def title_for_run(run_name: str) -> str:
    mapping = {
        "qwen06b_sft_iid_18k": "Qwen3-0.6B SFT Loss",
        "qwen06b_sft_iid_18k_stepweighted": "Qwen3-0.6B Step-Weighted SFT Loss",
        "qwen17b_sft_iid_18k": "Qwen3-1.7B SFT Loss",
        "qwen17b_base_sft_iid_18k": "Qwen3-1.7B-Base SFT Loss",
        "qwen4b_sft_iid_18k": "Qwen3-4B SFT Loss",
    }
    return mapping.get(run_name, run_name)


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def infer_total_steps(lines: Iterable[str]) -> Optional[int]:
    best = None
    for line in lines:
        match = STEP_RE.search(line)
        if not match:
            continue
        total = int(match.group(2))
        if best is None or total > best:
            best = total
    return best


def parse_log_dicts(log_text: str) -> Tuple[List[Dict], List[Dict]]:
    loss_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    for raw_line in log_text.splitlines():
        line = strip_ansi(raw_line).strip()
        if "{'loss':" not in line and "{'eval_loss':" not in line:
            continue
        match = LOSS_LINE_RE.search(line)
        if not match:
            continue
        payload = match.group(0)
        try:
            item = ast.literal_eval(payload)
        except Exception:
            continue
        if "loss" in item:
            loss_rows.append(item)
        elif "eval_loss" in item:
            eval_rows.append(item)
    return loss_rows, eval_rows


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def reconstruct_csv_from_log(log_path: Path, out_csv: Path) -> None:
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = log_text.splitlines()
    loss_rows, eval_rows = parse_log_dicts(log_text)
    if not loss_rows and not eval_rows:
        raise SystemExit(f"No loss entries found in {log_path}")

    epochs = []
    for row in loss_rows + eval_rows:
        epoch = to_float(row.get("epoch"))
        if epoch is not None:
            epochs.append(epoch)
    max_epoch = max(epochs) if epochs else None
    total_steps = infer_total_steps(lines)
    steps_per_epoch = None
    if total_steps is not None and max_epoch and max_epoch > 0:
        steps_per_epoch = total_steps / max_epoch

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "eval_loss"])
        writer.writeheader()

        rows = []
        for row in loss_rows:
            epoch = to_float(row.get("epoch"))
            loss = to_float(row.get("loss"))
            if epoch is None or loss is None:
                continue
            step = epoch * steps_per_epoch if steps_per_epoch is not None else len(rows) + 1
            rows.append({"step": f"{step:.4f}", "epoch": f"{epoch:.6f}", "loss": f"{loss:.6f}", "eval_loss": ""})

        for row in eval_rows:
            epoch = to_float(row.get("epoch"))
            eval_loss = to_float(row.get("eval_loss"))
            if epoch is None or eval_loss is None:
                continue
            step = epoch * steps_per_epoch if steps_per_epoch is not None else len(rows) + 1
            rows.append({"step": f"{step:.4f}", "epoch": f"{epoch:.6f}", "loss": "", "eval_loss": f"{eval_loss:.6f}"})

        rows.sort(key=lambda r: float(r["step"]))
        writer.writerows(rows)


def generate_plot(csv_path: Path, out_path: Path, title: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.plot_loss_curve",
        "--csv",
        str(csv_path),
        "--out",
        str(out_path),
        "--title",
        title,
    ]
    subprocess.run(cmd, check=True)


def build_run(run_name: str, outdir: Path) -> Dict[str, str]:
    csv_out = outdir / f"{run_name}_loss_curve.csv"
    svg_out = outdir / f"{run_name}_loss_curve.svg"
    html_out = outdir / f"{run_name}_loss_curve.html"

    csv_src = existing_csv_for_run(run_name)
    if csv_src is not None:
        copy_csv(csv_src, csv_out)
        source = str(csv_src)
    else:
        log_src = existing_log_for_run(run_name)
        if log_src is None:
            raise FileNotFoundError(f"No loss curve CSV or training log found for {run_name}")
        reconstruct_csv_from_log(log_src, csv_out)
        source = str(log_src)

    title = title_for_run(run_name)
    generate_plot(csv_out, svg_out, title)
    generate_plot(csv_out, html_out, title)

    return {
        "run": run_name,
        "source": source,
        "csv": str(csv_out),
        "svg": str(svg_out),
        "html": str(html_out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/loss_curves")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_name in RUNS:
        try:
            row = build_run(run_name, outdir)
            rows.append(row)
            print(f"built {run_name}")
        except FileNotFoundError as e:
            print(f"skip {run_name}: {e}")

    if rows:
        index_path = outdir / "INDEX.csv"
        with index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "source", "csv", "svg", "html"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {index_path}")


if __name__ == "__main__":
    main()
