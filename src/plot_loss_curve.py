import argparse
import csv
import html
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


def _svg_polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(out_path: Path, title: str, train_x, train_y, eval_x, eval_y):
    width, height = 900, 540
    left, right, top, bottom = 80, 30, 60, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = list(train_x) + list(eval_x)
    ys = list(train_y) + list(eval_y)
    if not xs or not ys:
        raise SystemExit("No plottable values available for SVG output.")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def sx(x):
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y):
        return top + plot_h - ((y - y_min) / (y_max - y_min)) * plot_h

    def ticks(lo, hi, n=5):
        step = (hi - lo) / max(n - 1, 1)
        return [lo + i * step for i in range(n)]

    train_points = [(sx(x), sy(y)) for x, y in zip(train_x, train_y)]
    eval_points = [(sx(x), sy(y)) for x, y in zip(eval_x, eval_y)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.axis { stroke: #374151; stroke-width: 1.5; }',
        '.train { fill: none; stroke: #2563eb; stroke-width: 2.5; }',
        '.eval { fill: none; stroke: #dc2626; stroke-width: 2.5; }',
        '.tick { font-size: 12px; }',
        '.title { font-size: 20px; font-weight: 600; }',
        '.legend { font-size: 13px; }',
        '</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]

    for xv in ticks(x_min, x_max):
        xpix = sx(xv)
        parts.append(f'<line x1="{xpix:.2f}" y1="{top}" x2="{xpix:.2f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(
            f'<text x="{xpix:.2f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{xv:.0f}</text>'
        )
    for yv in ticks(y_min, y_max):
        ypix = sy(yv)
        parts.append(f'<line x1="{left}" y1="{ypix:.2f}" x2="{left + plot_w}" y2="{ypix:.2f}" class="grid"/>')
        parts.append(
            f'<text x="{left - 10}" y="{ypix + 4:.2f}" text-anchor="end" class="tick">{yv:.2f}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(
        f'<text x="{left + plot_w/2:.2f}" y="{height - 15}" text-anchor="middle">Step</text>'
    )
    parts.append(
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20 {top + plot_h/2:.2f})" text-anchor="middle">Loss</text>'
    )

    if train_points:
        parts.append(f'<polyline points="{_svg_polyline(train_points)}" class="train"/>')
    if eval_points:
        parts.append(f'<polyline points="{_svg_polyline(eval_points)}" class="eval"/>')

    legend_x = left + 10
    legend_y = top + 10
    if train_points:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" class="train"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" class="legend">train_loss</text>')
        legend_y += 22
    if eval_points:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" class="eval"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" class="legend">eval_loss</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved SVG plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to loss_curve.csv or training_log.csv")
    ap.add_argument("--out", type=str, required=True, help="Output image path, e.g. outputs/loss.png")
    ap.add_argument("--title", type=str, default="Training Curve")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        plt = None
        mpl_error = e

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

    if plt is None:
        if out_path.suffix.lower() not in {".svg", ".html"}:
            out_path = out_path.with_suffix(".svg")
        if out_path.suffix.lower() == ".html":
            svg_path = out_path.with_suffix(".svg")
            write_svg(svg_path, args.title, train_x, train_y, eval_x, eval_y)
            out_path.write_text(
                "<html><body style='margin:0;padding:24px;background:#f8fafc'>"
                f"<img src='{html.escape(svg_path.name)}' style='max-width:100%;height:auto'/>"
                "</body></html>",
                encoding="utf-8",
            )
            print(
                f"matplotlib not available; wrote HTML wrapper to {out_path} and SVG to {svg_path}. "
                f"Original error: {mpl_error}"
            )
        else:
            write_svg(out_path, args.title, train_x, train_y, eval_x, eval_y)
            print(f"matplotlib not available; used SVG fallback. Original error: {mpl_error}")
        return

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
