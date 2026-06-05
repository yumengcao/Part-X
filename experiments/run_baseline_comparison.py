from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import BaselineConfig, make_baseline  # noqa: E402
from baselines.utils import make_sobol_candidates  # noqa: E402
from evaluation.dense_grid import make_evaluation_set  # noqa: E402
from evaluation.quantile_estimator import estimate_quantile_threshold, make_vectorized_function  # noqa: E402
from experiments.plot_baseline_comparison import plot_benchmark_results  # noqa: E402
from experiments.utils.benchmark_registry import (  # noqa: E402
    default_bounds_for,
    get_builtin_benchmark,
    infer_dimension,
)
from experiments.utils.result_io import ensure_dir, write_csv, write_json, write_text  # noqa: E402


METRIC_FIELDS = [
    "misclassification_volume",
    "undecided_volume",
    "classified_volume",
    "maintained_classified_volume",
    "symmetric_difference",
    "boundary_f1",
    "hausdorff_distance_2d",
    "runtime",
    "n_active_regions",
    "n_leaves",
    "n_samples",
    "budget_used",
]

ITER_FIELD_ORDER = [
    "benchmark",
    "seed",
    "method",
    "status",
    "iteration",
    "budget_used",
    "n_samples",
    *METRIC_FIELDS,
]

DEFAULT_METHODS = [
    "random_gp",
    "sobol_gp",
    "straddle",
    "gp_lse",
    "truvar",
    "sur",
    "upart_wrapper",
    "pbnb_wrapper",
]

DEFAULT_SEED_SETS = {
    "debug": [0],
    "standard": [0, 1, 2, 3, 4],
    "final": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        import json

        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to read YAML configs. Install pyyaml or use a JSON config."
        ) from exc
    data = yaml.safe_load(text)
    return {} if data is None else dict(data)


def run_comparison_from_config(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    methods: Optional[Iterable[str]] = None,
    benchmarks: Optional[Iterable[str]] = None,
    benchmark_groups: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    seed_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Run the unified baseline comparison and return generated output paths."""

    config = (
        load_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else dict(config_or_path)
    )

    output_root = ensure_dir(config.get("output_root", "experiments/outputs/baseline_comparison"))
    global_cfg = dict(config.get("global", {}))
    selected_methods = _select_methods(config.get("methods", DEFAULT_METHODS), methods)
    selected_benchmarks = _select_benchmarks(
        config.get("benchmarks", {}),
        benchmarks,
        benchmark_groups=benchmark_groups,
        group_config=config.get("benchmark_groups", {}),
    )
    selected_seeds = _select_seeds(config, seeds=seeds, seed_mode=seed_mode)

    results: dict[str, Any] = {"output_root": str(output_root), "benchmarks": {}}
    for benchmark_name, benchmark_cfg in selected_benchmarks.items():
        benchmark_dir = ensure_dir(output_root / benchmark_name)
        benchmark_iter_rows: list[dict[str, Any]] = []
        benchmark_final_rows: list[dict[str, Any]] = []

        benchmark_fn, bounds = build_benchmark(benchmark_name, benchmark_cfg)

        for seed in selected_seeds:
            seed_dir = ensure_dir(benchmark_dir / f"seed_{seed}")
            threshold, threshold_metadata = resolve_benchmark_threshold(
                benchmark_name,
                benchmark_fn,
                bounds,
                benchmark_cfg,
                global_cfg,
                seed,
            )
            write_json(threshold_metadata, seed_dir / "threshold_metadata.json")
            shared_sets = _make_shared_sets(bounds, global_cfg, benchmark_cfg, seed)
            for method_name, method_cfg in selected_methods:
                method_dir = ensure_dir(seed_dir / method_name)
                metadata = {
                    "benchmark": benchmark_name,
                    "method": method_name,
                    "seed": seed,
                    "bounds": bounds,
                    "threshold": threshold,
                    "threshold_metadata": threshold_metadata,
                    "config": {
                        "global": global_cfg,
                        "benchmark": benchmark_cfg,
                        "method": method_cfg,
                    },
                }
                write_json(metadata, method_dir / "metadata.json")

                try:
                    iter_rows, final_row = run_one_method(
                        benchmark_name=benchmark_name,
                        benchmark_fn=benchmark_fn,
                        bounds=bounds,
                        threshold=threshold,
                        method_name=method_name,
                        method_cfg=method_cfg,
                        global_cfg=global_cfg,
                        benchmark_cfg=benchmark_cfg,
                        seed=seed,
                        output_dir=method_dir,
                        shared_sets=shared_sets,
                    )
                    benchmark_iter_rows.extend(iter_rows)
                    benchmark_final_rows.append(final_row)
                except Exception as exc:  # noqa: BLE001 - deliberate failure isolation
                    status = "skipped" if isinstance(exc, NotImplementedError) else "failed"
                    error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    write_text(error_text, method_dir / "error.log")
                    final_row = _failure_row(
                        benchmark_name,
                        method_name,
                        seed,
                        status=status,
                        error=str(exc),
                    )
                    write_json(final_row, method_dir / "final_metrics.json")
                    benchmark_final_rows.append(final_row)

        write_csv(benchmark_iter_rows, benchmark_dir / "summary_iter_metrics.csv", fieldnames=None)
        write_csv(benchmark_final_rows, benchmark_dir / "summary_final_metrics.csv", fieldnames=None)
        plot_paths = plot_benchmark_results(benchmark_dir)
        results["benchmarks"][benchmark_name] = {
            "benchmark_dir": str(benchmark_dir),
            "summary_iter_metrics": str(benchmark_dir / "summary_iter_metrics.csv"),
            "summary_final_metrics": str(benchmark_dir / "summary_final_metrics.csv"),
            "plots": [str(path) for path in plot_paths],
        }

    return results


def run_one_method(
    *,
    benchmark_name: str,
    benchmark_fn: Callable[..., Any],
    bounds: list[tuple[float, float]],
    threshold: float,
    method_name: str,
    method_cfg: Mapping[str, Any],
    global_cfg: Mapping[str, Any],
    benchmark_cfg: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    shared_sets: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one method for one benchmark/seed and save method-level outputs."""

    config = _baseline_config(bounds, threshold, global_cfg, benchmark_cfg, method_cfg, seed, shared_sets)
    baseline = _make_method(method_name, benchmark_fn, config, global_cfg, benchmark_cfg, method_cfg, output_dir)
    max_iterations = int(method_cfg.get("max_iterations", benchmark_cfg.get("max_iterations", global_cfg.get("max_iterations", 10))))

    iter_rows: list[dict[str, Any]] = []
    start = time.time()
    for _ in range(max_iterations):
        metrics = dict(baseline.run_one_iteration())
        elapsed = time.time() - start
        if metrics.get("runtime") is None:
            metrics["runtime"] = elapsed
        row = _normalize_metric_row(metrics, benchmark_name, method_name, seed, status="ok")
        iter_rows.append(row)
        if _budget_exhausted(metrics, config.total_budget):
            break

        # Full-run partition wrappers expose a single complete run rather than
        # an iterative acquisition loop.
        if method_name.lower() in {"upart", "upart_wrapper", "partx", "partx_wrapper", "partx_baseline"}:
            break

    if not iter_rows:
        raise RuntimeError(f"Method {method_name!r} produced no iteration metrics.")

    final_row = dict(iter_rows[-1])
    final_row["status"] = "ok"
    write_csv(iter_rows, output_dir / "iter_metrics.csv", fieldnames=None)
    write_json(final_row, output_dir / "final_metrics.json")
    return iter_rows, final_row


def build_benchmark(name: str, cfg: Mapping[str, Any]) -> tuple[Callable[..., Any], list[tuple[float, float]]]:
    """Build a benchmark callable and bounds from config."""

    if "callable" in cfg:
        benchmark = _load_callable(str(cfg["callable"]))
    else:
        benchmark = get_builtin_benchmark(str(cfg.get("function", name)))

    dimension = infer_dimension(name, cfg)
    bounds = cfg.get("bounds")
    if bounds is None:
        if dimension is None:
            raise ValueError(f"Benchmark {name!r} must define either bounds or dimension.")
        bounds = default_bounds_for(str(cfg.get("function", name)), dimension)
    return benchmark, [tuple(map(float, pair)) for pair in bounds]


def resolve_benchmark_threshold(
    name: str,
    benchmark: Callable[..., Any],
    bounds: list[tuple[float, float]],
    cfg: Mapping[str, Any],
    global_cfg: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve a fixed or quantile-estimated threshold for one benchmark/seed."""

    global_cfg = {} if global_cfg is None else global_cfg
    threshold_type = str(cfg.get("threshold_type", "fixed" if "threshold" in cfg else "fixed")).lower()
    if threshold_type == "fixed":
        threshold = float(cfg.get("threshold", 0.0))
        return threshold, {
            "benchmark": name,
            "threshold_type": "fixed",
            "threshold": threshold,
            "seed": seed,
        }
    if threshold_type != "quantile":
        raise ValueError("threshold_type must be either 'fixed' or 'quantile'.")

    quantile_level = float(cfg.get("quantile_level", cfg.get("delta", 0.1)))
    method = str(cfg.get("quantile_method", global_cfg.get("quantile_method", "sobol")))
    n_points = int(cfg.get("quantile_n_points", global_cfg.get("quantile_n_points", 8192)))
    grid_per_dim = cfg.get("quantile_grid_per_dim", global_cfg.get("quantile_grid_per_dim"))
    dim_grid_threshold = int(cfg.get("quantile_dim_grid_threshold", global_cfg.get("quantile_dim_grid_threshold", 3)))
    reference_threshold = cfg.get("reference_threshold")

    estimate = estimate_quantile_threshold(
        make_vectorized_function(benchmark, r_eval=1),
        bounds,
        delta=quantile_level,
        method=method,
        n_points=n_points,
        grid_per_dim=grid_per_dim,
        dim_grid_threshold=dim_grid_threshold,
        seed=seed,
        benchmark=name,
        function_module=str(cfg.get("callable", cfg.get("function", name))),
        reference_threshold=None if reference_threshold is None else float(reference_threshold),
    )
    metadata = estimate.to_dict()
    metadata["threshold_type"] = "quantile"
    return float(estimate.threshold), metadata


def _load_callable(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError("callable must be formatted as 'module:function' or 'module.function'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _baseline_config(
    bounds,
    threshold,
    global_cfg,
    benchmark_cfg,
    method_cfg,
    seed,
    shared_sets,
) -> BaselineConfig:
    total_budget = _first_present(method_cfg, benchmark_cfg, global_cfg, "total_budget", "recommended_budget")
    initial_samples = _first_present(method_cfg, benchmark_cfg, global_cfg, "initial_samples", "recommended_initial_samples")
    eval_points = _first_present(method_cfg, benchmark_cfg, global_cfg, "n_eval_points", "evaluation_set_size")
    return BaselineConfig(
        region=tuple(tuple(pair) for pair in bounds),
        threshold=float(method_cfg.get("threshold", threshold)),
        total_budget=None if total_budget is None else int(total_budget),
        initial_samples=int(20 if initial_samples is None else initial_samples),
        batch_size=int(method_cfg.get("batch_size", benchmark_cfg.get("batch_size", global_cfg.get("batch_size", 1)))),
        n_candidates=int(method_cfg.get("candidate_set_size", benchmark_cfg.get("candidate_set_size", global_cfg.get("candidate_set_size", 512)))),
        n_eval_points=int(1024 if eval_points is None else eval_points),
        beta=float(method_cfg.get("beta", benchmark_cfg.get("beta", global_cfg.get("beta", 2.0)))),
        seed=int(seed),
        candidate_set=shared_sets["candidate_set"],
        evaluation_set=shared_sets["evaluation_set"],
        fit_gp=bool(method_cfg.get("fit_gp", global_cfg.get("fit_gp", True))),
        evaluate_truth=bool(method_cfg.get("evaluate_truth", global_cfg.get("evaluate_truth", True))),
    )


def _make_method(method_name, benchmark_fn, config, global_cfg, benchmark_cfg, method_cfg, output_dir):
    key = method_name.lower()
    if key in {"upart", "upart_wrapper", "partx", "partx_wrapper", "partx_baseline"}:
        from baselines.wrappers import PartXBaseline, PartXWrapper

        sampling_method = str(method_cfg.get("sampling_method", benchmark_cfg.get("sampling_method", global_cfg.get("sampling_method", "BO"))))
        params = SimpleNamespace(
            region=list(config.region),
            method=sampling_method,
            budget=config.total_budget,
            initial_budget_per_region=max(1, int(config.initial_samples)),
            max_iterations=int(_first_present(method_cfg, benchmark_cfg, global_cfg, "max_iterations") or 10),
            seed=config.seed,
        )
        args = SimpleNamespace(
            out_csv=str(output_dir / "legacy_iter_stats.csv"),
            mis_mc_samples=int(method_cfg.get("mis_mc_samples", benchmark_cfg.get("mis_mc_samples", global_cfg.get("mis_mc_samples", 1000)))),
            threshold=config.threshold,
            h=None,
            split_rule=str(method_cfg.get("split_rule", "legacy")),
            partition_variant=None,
            no_guided_plots=True,
            guided_make_plots=False,
        )
        cls = PartXBaseline if key in {"upart", "upart_wrapper", "partx_baseline"} else PartXWrapper
        baseline = cls(
            benchmark_fn,
            config,
            sampling_method=sampling_method,
            params=params,
            args=args,
        )
        baseline.method_name = method_name
        return baseline

    return make_baseline(method_name, benchmark_fn, config)


def _make_shared_sets(bounds, global_cfg, benchmark_cfg, seed: int) -> dict[str, np.ndarray]:
    candidate_n = int(benchmark_cfg.get("candidate_set_size", global_cfg.get("candidate_set_size", 512)))
    eval_n = int(_first_present(benchmark_cfg, global_cfg, {}, "n_eval_points", "evaluation_set_size") or 1024)
    grid_size = int(benchmark_cfg.get("eval_grid_size", global_cfg.get("eval_grid_size", max(2, round(math.sqrt(eval_n))))))
    evaluation_type = str(benchmark_cfg.get("evaluation_type", "")).lower()
    if evaluation_type:
        prefer_grid = evaluation_type == "dense_grid"
    else:
        prefer_grid = bool(benchmark_cfg.get("prefer_grid_for_2d", global_cfg.get("prefer_grid_for_2d", True)))
    candidate_set = make_sobol_candidates(bounds, candidate_n, seed=seed)
    evaluation_set, _ = make_evaluation_set(
        bounds,
        grid_size=grid_size,
        n_sobol=eval_n,
        seed=seed + 10_000,
        prefer_grid_for_2d=prefer_grid,
    )
    return {"candidate_set": candidate_set, "evaluation_set": evaluation_set}


def _select_methods(config_methods, override: Optional[Iterable[str]]) -> list[tuple[str, dict[str, Any]]]:
    if override is not None:
        return [(str(method), {}) for method in override]

    selected = []
    for entry in config_methods:
        if isinstance(entry, str):
            selected.append((entry, {}))
            continue
        if not bool(entry.get("enabled", True)):
            continue
        name = str(entry["name"])
        cfg = {key: value for key, value in entry.items() if key != "name"}
        selected.append((name, cfg))
    return selected


def _select_benchmarks(
    config_benchmarks,
    override: Optional[Iterable[str]],
    *,
    benchmark_groups: Optional[Iterable[str]] = None,
    group_config: Optional[Mapping[str, Iterable[str]]] = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(config_benchmarks, Mapping):
        raise ValueError("config['benchmarks'] must be a mapping.")

    requested = None if override is None else {str(name) for name in override}
    if requested is None and benchmark_groups is not None:
        group_config = {} if group_config is None else group_config
        requested = set()
        groups = [str(group).lower() for group in benchmark_groups]
        if "all" in groups:
            requested = set(config_benchmarks.keys())
        else:
            for group in groups:
                if group not in group_config:
                    raise KeyError(f"Unknown benchmark group {group!r}.")
                requested.update(str(name) for name in group_config[group])
    selected = {}
    for name, cfg in config_benchmarks.items():
        cfg = {} if cfg is None else dict(cfg)
        if requested is not None and name not in requested:
            continue
        if requested is None and not bool(cfg.get("enabled", True)):
            continue
        selected[str(name)] = cfg
    if requested is not None:
        missing = requested.difference(selected)
        if missing:
            raise KeyError(f"Requested benchmark(s) not found or disabled in config: {sorted(missing)}")
    return selected


def _select_seeds(config, *, seeds: Optional[Iterable[int]], seed_mode: Optional[str]) -> list[int]:
    if seeds is not None:
        return [int(seed) for seed in seeds]
    mode = seed_mode or config.get("seed_mode")
    seed_sets = dict(DEFAULT_SEED_SETS)
    seed_sets.update(config.get("seed_sets", {}))
    if mode is not None:
        key = str(mode).lower()
        if key not in seed_sets:
            raise KeyError(f"Unknown seed_mode {mode!r}. Available: {sorted(seed_sets)}")
        return [int(seed) for seed in seed_sets[key]]
    return [int(seed) for seed in config.get("seeds", [0])]


def _first_present(*mappings_and_keys):
    mappings = [item for item in mappings_and_keys if isinstance(item, Mapping)]
    keys = [item for item in mappings_and_keys if not isinstance(item, Mapping)]
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _normalize_metric_row(metrics, benchmark, method, seed, status):
    row = dict(metrics)
    row.update(
        {
            "benchmark": benchmark,
            "method": method,
            "seed": int(seed),
            "status": status,
        }
    )
    if row.get("budget_used") is None:
        row["budget_used"] = row.get("n_samples")
    return row


def _failure_row(benchmark, method, seed, *, status, error):
    return {
        "benchmark": benchmark,
        "method": method,
        "seed": int(seed),
        "status": status,
        "error": error,
        "iteration": None,
        **{field: None for field in METRIC_FIELDS},
    }


def _budget_exhausted(metrics, total_budget) -> bool:
    if total_budget is None:
        return False
    used = metrics.get("budget_used", metrics.get("n_samples"))
    if used is None:
        return False
    return int(used) >= int(total_budget)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified level-set baseline comparisons.")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/baseline_comparison.yaml")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--benchmark", "--benchmarks", nargs="+", default=None, dest="benchmarks")
    parser.add_argument(
        "--benchmark-group",
        "--benchmark-groups",
        nargs="+",
        default=None,
        dest="benchmark_groups",
        help="Benchmark group(s) to run, e.g. 2d 3d 6d 10d all.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed-mode", choices=["debug", "standard", "final"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_comparison_from_config(
        args.config,
        methods=args.methods,
        benchmarks=args.benchmarks,
        benchmark_groups=args.benchmark_groups,
        seeds=args.seeds,
        seed_mode=args.seed_mode,
    )
    for benchmark, info in results["benchmarks"].items():
        print(f"{benchmark}: {info['benchmark_dir']}")


if __name__ == "__main__":
    main()
