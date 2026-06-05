from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


CURVE_METRICS = [
    "misclassification_volume",
    "undecided_volume",
    "classified_volume",
    "maintained_classified_volume",
    "boundary_f1",
    "runtime",
]


def plot_benchmark_results(
    benchmark_dir: str | Path,
    *,
    metrics: Optional[Iterable[str]] = None,
    std_shading: bool = True,
) -> list[Path]:
    """Generate comparison plots for one benchmark output directory."""

    import matplotlib.pyplot as plt

    benchmark_path = Path(benchmark_dir)
    summary_path = benchmark_path / "summary_iter_metrics.csv"
    if not summary_path.exists():
        return []

    df = pd.read_csv(summary_path)
    if df.empty:
        return []

    plot_dir = benchmark_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for metric in list(metrics or CURVE_METRICS):
        if metric not in df.columns:
            continue
        path = _plot_metric_curve(df, metric, plot_dir, std_shading=std_shading)
        if path is not None:
            generated.append(path)

    if "n_leaves" in df.columns:
        path = _plot_metric_curve(
            df,
            "n_leaves",
            plot_dir,
            x_col="iteration",
            title_suffix="vs iteration",
            std_shading=std_shading,
        )
        if path is not None:
            generated.append(path)

    return generated


def _plot_metric_curve(
    df: pd.DataFrame,
    metric: str,
    plot_dir: Path,
    *,
    x_col: str = "budget_used",
    title_suffix: str = "vs budget",
    std_shading: bool = True,
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    if x_col not in df.columns:
        return None

    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work = work[np.isfinite(work[metric]) & np.isfinite(work[x_col])]
    if work.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, group in work.groupby("method", sort=True):
        group = group.sort_values(["iteration", x_col])
        grouped = group.groupby("iteration", as_index=False).agg(
            x=(x_col, "mean"),
            y=(metric, "mean"),
            y_std=(metric, "std"),
        )
        grouped["y_std"] = grouped["y_std"].fillna(0.0)
        ax.plot(grouped["x"], grouped["y"], marker="o", linewidth=2, label=str(method))
        if std_shading and len(group["seed"].dropna().unique()) > 1:
            lower = grouped["y"] - grouped["y_std"]
            upper = grouped["y"] + grouped["y_std"]
            ax.fill_between(grouped["x"], lower, upper, alpha=0.15)

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} {title_suffix}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = plot_dir / f"{metric}_{title_suffix.replace(' ', '_')}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot unified baseline comparison outputs.")
    parser.add_argument("benchmark_dir", type=Path, help="Benchmark output directory.")
    parser.add_argument("--no-std-shading", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = plot_benchmark_results(args.benchmark_dir, std_shading=not args.no_std_shading)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
