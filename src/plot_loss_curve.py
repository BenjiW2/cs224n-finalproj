import argparse
import csv
from pathlib import Path


def read_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to loss_curve.csv or training_log.csv")
    ap.add_argument("--out", type=str, required=True, help="Output image path, e.g. outputs/loss.png")
    ap.add_argument("--title", type=str, default="Training Curve")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`. "
            f"Original error: {e}"
        )

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    train_x, train_y = [], []
    eval_x, eval_y = [], []

    for row in rows:
        step = _to_float(row.get("step"))
        if step is None:
            continue
        loss = _to_float(row.get("loss"))
        eval_loss = _to_float(row.get("eval_loss"))
        if loss is not None:
            train_x.append(step)
            train_y.append(loss)
        if eval_loss is not None:
            eval_x.append(step)
            eval_y.append(eval_loss)

    if not train_x and not eval_x:
        raise SystemExit(f"No plottable loss values found in {args.csv}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if train_x:
        plt.plot(train_x, train_y, marker="o", linewidth=1.5, markersize=3, label="train_loss")
    if eval_x:
        plt.plot(eval_x, eval_y, marker="s", linewidth=1.5, markersize=4, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    if train_x and eval_x:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
