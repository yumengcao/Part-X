import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments.run_baseline_comparison import (
    build_benchmark,
    load_config,
    resolve_benchmark_threshold,
    run_comparison_from_config,
    _select_benchmarks,
)
from experiments.utils.benchmark_registry import get_benchmark_spec


_MISSING_DEPS = [
    name
    for name in ("torch", "botorch", "gpytorch")
    if importlib.util.find_spec(name) is None
]

REQUIRED_BENCHMARK_GROUPS = {
    "2d": [
        "rosenbrock2",
        "himmelblau",
        "branin2",
        "discontinuous2d",
        "shifted_sine2",
    ],
    "3d": ["rosenbrock3", "ackley3", "levy3"],
    "6d": ["rosenbrock6", "ackley6", "levy6"],
    "10d": ["rosenbrock10", "ackley10", "levy10"],
}
REQUIRED_BENCHMARKS = [
    name
    for group_names in REQUIRED_BENCHMARK_GROUPS.values()
    for name in group_names
]


def toy2d(X, r=1):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        return arr[:, 0] + arr[:, 1] - 1.0
    value = float(arr[0] + arr[1] - 1.0)
    return [value for _ in range(r)]


class TestBenchmarkSuiteConfig(unittest.TestCase):
    def test_baseline_config_covers_required_multidimensional_suite(self):
        config = load_config("experiments/configs/baseline_comparison.yaml")
        self._assert_required_suite_config(config)

    def test_debug_config_covers_required_multidimensional_suite(self):
        config = load_config("experiments/configs/baseline_comparison_debug.yaml")
        self._assert_required_suite_config(config)

    def test_debug_config_builds_benchmarks_and_thresholds(self):
        config = load_config("experiments/configs/baseline_comparison_debug.yaml")
        benchmarks = _select_benchmarks(
            config["benchmarks"],
            None,
            benchmark_groups=["all"],
            group_config=config["benchmark_groups"],
        )
        self.assertEqual(set(benchmarks), set(REQUIRED_BENCHMARKS))

        for name, benchmark_cfg in benchmarks.items():
            spec = get_benchmark_spec(name)
            fn, bounds = build_benchmark(name, benchmark_cfg)
            threshold, metadata = resolve_benchmark_threshold(
                name,
                fn,
                bounds,
                benchmark_cfg,
                config["global"],
                seed=0,
            )
            center = np.asarray([(low + high) / 2.0 for low, high in bounds])
            value = np.asarray(fn(center, 1), dtype=float).reshape(-1)
            self.assertGreater(value.size, 0)
            self.assertEqual(len(bounds), spec.dimension)
            self.assertIsInstance(threshold, float)
            self.assertIn("threshold_type", metadata)

        for group, expected_names in REQUIRED_BENCHMARK_GROUPS.items():
            selected = _select_benchmarks(
                config["benchmarks"],
                None,
                benchmark_groups=[group],
                group_config=config["benchmark_groups"],
            )
            self.assertEqual(set(selected), set(expected_names))

    def _assert_required_suite_config(self, config):
        groups = config["benchmark_groups"]
        benchmarks = config["benchmarks"]
        for group, expected_names in REQUIRED_BENCHMARK_GROUPS.items():
            self.assertEqual(groups[group], expected_names)

        self.assertEqual(groups["all"], REQUIRED_BENCHMARKS)
        self.assertGreaterEqual(len(groups["2d"]), 3)
        self.assertGreaterEqual(len(groups["3d"]), 3)
        self.assertGreaterEqual(len(groups["6d"]), 3)
        self.assertGreaterEqual(len(groups["10d"]), 3)
        self.assertIn("discontinuous2d", groups["2d"])

        for name in REQUIRED_BENCHMARKS:
            self.assertIn(name, benchmarks)
            benchmark_cfg = benchmarks[name]
            spec = get_benchmark_spec(name)
            self.assertTrue(benchmark_cfg.get("enabled", True))
            self.assertEqual(benchmark_cfg["group"], f"{spec.dimension}d")
            self.assertEqual(int(benchmark_cfg["dimension"]), spec.dimension)
            self.assertEqual(
                [tuple(pair) for pair in benchmark_cfg["bounds"]],
                list(spec.bounds),
            )
            self.assertEqual(benchmark_cfg["threshold_type"], spec.threshold_type)
            if spec.threshold_type == "fixed":
                self.assertAlmostEqual(float(benchmark_cfg["threshold"]), spec.threshold)
            else:
                self.assertAlmostEqual(
                    float(benchmark_cfg["quantile_level"]),
                    spec.quantile_level,
                )


@unittest.skipIf(
    bool(_MISSING_DEPS),
    "requires optional BoTorch GP stack: " + ", ".join(_MISSING_DEPS),
)
class TestBaselineComparisonRunner(unittest.TestCase):

    def test_runner_smoke_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            config = {
                "output_root": str(output_root),
                "seeds": [0],
                "global": {
                    "max_iterations": 2,
                    "total_budget": 6,
                    "initial_samples": 3,
                    "batch_size": 2,
                    "candidate_set_size": 8,
                    "n_eval_points": 16,
                    "eval_grid_size": 4,
                    "beta": 2.0,
                    "evaluate_truth": True,
                    "fit_gp": True,
                },
                "methods": [
                    {"name": "random_gp"},
                    {"name": "sobol_gp"},
                    {"name": "missing_method"},
                ],
                "benchmarks": {
                    "toy2d": {
                        "callable": "tests.test_baseline_comparison_runner:toy2d",
                        "bounds": [[0.0, 1.0], [0.0, 1.0]],
                        "threshold": 0.0,
                    }
                },
            }
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            result = run_comparison_from_config(config_path)

            benchmark_dir = output_root / "toy2d"
            self.assertIn("toy2d", result["benchmarks"])
            self.assertTrue((benchmark_dir / "summary_iter_metrics.csv").exists())
            self.assertTrue((benchmark_dir / "summary_final_metrics.csv").exists())

            for method in ("random_gp", "sobol_gp"):
                method_dir = benchmark_dir / "seed_0" / method
                self.assertTrue((method_dir / "iter_metrics.csv").exists())
                self.assertTrue((method_dir / "final_metrics.json").exists())

            failed_dir = benchmark_dir / "seed_0" / "missing_method"
            self.assertTrue((failed_dir / "error.log").exists())
            self.assertTrue((failed_dir / "final_metrics.json").exists())

            summary = (benchmark_dir / "summary_final_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("random_gp", summary)
            self.assertIn("sobol_gp", summary)
            self.assertIn("missing_method", summary)
            self.assertIn("failed", summary)


if __name__ == "__main__":
    unittest.main()
