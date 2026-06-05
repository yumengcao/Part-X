from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BenchmarkSpec:
    """Canonical metadata for an experiment benchmark variant."""

    name: str
    family: str
    dimension: int
    bounds: tuple[tuple[float, float], ...]
    threshold_type: str
    threshold: float | None = None
    quantile_level: float | None = None


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "rosenbrock2": BenchmarkSpec(
        name="rosenbrock2",
        family="rosenbrock",
        dimension=2,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        threshold_type="fixed",
        threshold=9.93098,
    ),
    "himmelblau": BenchmarkSpec(
        name="himmelblau",
        family="himmelblau",
        dimension=2,
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        threshold_type="fixed",
        threshold=0.0,
    ),
    "branin2": BenchmarkSpec(
        name="branin2",
        family="branin",
        dimension=2,
        bounds=((-5.0, 10.0), (0.0, 15.0)),
        threshold_type="fixed",
        threshold=0.0,
    ),
    "discontinuous2d": BenchmarkSpec(
        name="discontinuous2d",
        family="discontinuous",
        dimension=2,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        threshold_type="fixed",
        threshold=0.0,
    ),
    "shifted_sine2": BenchmarkSpec(
        name="shifted_sine2",
        family="shifted_sine",
        dimension=2,
        bounds=((-60.0, 60.0), (-60.0, 60.0)),
        threshold_type="fixed",
        threshold=0.0,
    ),
    "rosenbrock3": BenchmarkSpec(
        name="rosenbrock3",
        family="rosenbrock",
        dimension=3,
        bounds=((-2.0, 2.0),) * 3,
        threshold_type="fixed",
        threshold=92.9442,
    ),
    "ackley3": BenchmarkSpec(
        name="ackley3",
        family="ackley",
        dimension=3,
        bounds=((-32.768, 32.768),) * 3,
        threshold_type="fixed",
        threshold=19.3185,
    ),
    "levy3": BenchmarkSpec(
        name="levy3",
        family="levy",
        dimension=3,
        bounds=((-10.0, 10.0),) * 3,
        threshold_type="fixed",
        threshold=6.54471,
    ),
    "rosenbrock6": BenchmarkSpec(
        name="rosenbrock6",
        family="rosenbrock",
        dimension=6,
        bounds=((-2.0, 2.0),) * 6,
        threshold_type="fixed",
        threshold=633.085,
    ),
    "ackley6": BenchmarkSpec(
        name="ackley6",
        family="ackley",
        dimension=6,
        bounds=((-32.768, 32.768),) * 6,
        threshold_type="quantile",
        quantile_level=0.1,
    ),
    "levy6": BenchmarkSpec(
        name="levy6",
        family="levy",
        dimension=6,
        bounds=((-10.0, 10.0),) * 6,
        threshold_type="quantile",
        quantile_level=0.1,
    ),
    "rosenbrock10": BenchmarkSpec(
        name="rosenbrock10",
        family="rosenbrock",
        dimension=10,
        bounds=((-2.0, 2.0),) * 10,
        threshold_type="quantile",
        quantile_level=0.1,
    ),
    "ackley10": BenchmarkSpec(
        name="ackley10",
        family="ackley",
        dimension=10,
        bounds=((-32.768, 32.768),) * 10,
        threshold_type="quantile",
        quantile_level=0.1,
    ),
    "levy10": BenchmarkSpec(
        name="levy10",
        family="levy",
        dimension=10,
        bounds=((-10.0, 10.0),) * 10,
        threshold_type="quantile",
        quantile_level=0.1,
    ),
}


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    """Return canonical metadata for a benchmark variant."""

    key = name.lower()
    if key not in BENCHMARK_SPECS:
        raise KeyError(f"No benchmark spec registered for {name!r}.")
    return BENCHMARK_SPECS[key]


def list_benchmark_specs() -> dict[str, BenchmarkSpec]:
    """Return all registered benchmark specs keyed by canonical name."""

    return dict(BENCHMARK_SPECS)


def get_builtin_benchmark(name: str) -> Callable[..., Any]:
    """Return a project or registry benchmark callable by name."""

    key = name.lower()
    if key in {"himmelblau", "himmelblau2", "himm"}:
        from testfunction_himm import fun

        return fun
    if key.startswith("rosenbrock"):
        from testfunction_rosenbrock import testfunction_rosenbrock_d

        return testfunction_rosenbrock_d
    if key in {"branin2", "branin"}:
        from testfunction_branin_2d import testfunction_branin

        return testfunction_branin
    if key.startswith("shifted_sine") or key.startswith("shifted_sin"):
        from testfunction_shifted_sin import testfunction_shifted_sin_nd

        return testfunction_shifted_sin_nd
    if key.startswith("ackley"):
        from testfunction_ackley import testfunction_ackley

        return testfunction_ackley
    if key.startswith("levy"):
        return levy_nd
    if key in {"discontinuous2d", "nonstationary2d"}:
        return discontinuous2d
    raise KeyError(f"No built-in benchmark preset for {name!r}; provide callable in config.")


def levy_nd(X: Sequence[float] | np.ndarray, r: int = 1, mu: float = 0.0, sigma: float = 0.0):
    """Standard d-dimensional Levy benchmark on a bounded rectangular domain."""

    x = np.asarray(X, dtype=float)
    if x.ndim == 2:
        return np.asarray([_levy_scalar(row) for row in x], dtype=float)
    value = _levy_scalar(x.reshape(-1))
    noise = np.random.normal(mu, sigma, int(r))
    return (value + noise).tolist()


def discontinuous2d(X: Sequence[float] | np.ndarray, r: int = 1):
    """Simple discontinuous 2D level-set benchmark for registry-only experiments."""

    x = np.asarray(X, dtype=float)
    if x.ndim == 2:
        return np.asarray([_discontinuous2d_scalar(row) for row in x], dtype=float)
    return [float(_discontinuous2d_scalar(x.reshape(-1))) for _ in range(int(r))]


def infer_dimension(name: str, cfg: Mapping[str, Any]) -> int | None:
    """Infer benchmark dimension from config, name suffix, or bounds."""

    if "dimension" in cfg:
        return int(cfg["dimension"])
    if "bounds" in cfg:
        return len(cfg["bounds"])

    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else None


def default_bounds_for(name: str, dimension: int) -> list[tuple[float, float]]:
    """Return default bounds for known benchmark families."""

    key = name.lower()
    if key.startswith("rosenbrock"):
        return [(-2.0, 2.0)] * int(dimension)
    if key.startswith("ackley"):
        return [(-32.768, 32.768)] * int(dimension)
    if key.startswith("levy"):
        return [(-10.0, 10.0)] * int(dimension)
    if key.startswith("shifted_sine") or key.startswith("shifted_sin"):
        return [(-60.0, 60.0)] * int(dimension)
    if key in {"himmelblau", "himmelblau2", "himm"}:
        return [(-5.0, 5.0), (-5.0, 5.0)]
    if key in {"branin", "branin2"}:
        return [(-5.0, 10.0), (0.0, 15.0)]
    if key in {"discontinuous2d", "nonstationary2d"}:
        return [(-1.0, 1.0), (-1.0, 1.0)]
    raise KeyError(f"No default bounds are known for {name!r}.")


def _levy_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Levy requires at least one dimension.")
    w = 1.0 + (x - 1.0) / 4.0
    term1 = math.sin(math.pi * w[0]) ** 2
    if x.size > 1:
        middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * w[:-1] + 1.0) ** 2))
    else:
        middle = 0.0
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return float(term1 + middle + term3)


def _discontinuous2d_scalar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 2:
        raise ValueError("discontinuous2d expects a 2D input.")
    jump = 0.75 if x[0] >= 0.0 else -0.25
    ripple = 0.15 * math.sin(8.0 * x[1])
    return float(x[0] + 0.5 * x[1] + jump + ripple)
