Part-X Demo: Stochastic Search-Based Test Generation with Probabilistic Guarantees
==========================================================================

This repository provides a simplified, educational demo inspired by the published Part-X algorithm:

Giulia Pedrielli, Tanmay Khandait, Yumeng Cao, Quinn Thibeault, Hao Huang, Mauricio Castillo-Effen, and Georgios Fainekos, "Part-X: A Family of Stochastic Algorithms for Search-Based Test Generation with Probabilistic Guarantees," IEEE Transactions on Automation Science and Engineering, 2023.

This demo illustrates core ideas from Part-X in a compact, readable implementation. It is not the original research code and does not reproduce the full theoretical guarantees — it is intended for teaching and portfolio purposes only.

Features
--------

- Simple 2D synthetic robustness function on the domain [0,1]^2.
- Region partitioning (rectangular), sampling, and refinement.
- Simple binomial-based falsification probability estimation with confidence intervals.
- Visualization of sampled points, partitions, and estimated risky regions.

Quick start
-----------

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo:

```bash
python examples/run_partx_demo.py
```

This will run a short Part-X-style loop and save figures to `figures/iteration_1.png`, `figures/iteration_2.png`, etc.

Interpretation
--------------

- Contours show the true robustness function. Negative values indicate falsification (unsafe).
- Points are sampled and colored by safe (blue) or falsifying (red).
- Rectangles show the current partition of the search space. Regions are shaded/outlined based on estimated falsification probability.

Disclaimer
----------

This repository demonstrates simplified Part-X-style ideas. The probability estimates and heuristics used here are educational approximations and should not be treated as reproductions of the published algorithm or its formal guarantees.

CITATION
--------

If you use this demo or build on the published algorithm, please cite:

Giulia Pedrielli, Tanmay Khandait, Yumeng Cao, Quinn Thibeault, Hao Huang, Mauricio Castillo-Effen, and Georgios Fainekos, "Part-X: A Family of Stochastic Algorithms for Search-Based Test Generation with Probabilistic Guarantees," IEEE Transactions on Automation Science and Engineering, 2023.

Author
------

- Yumeng Cao (demo implementation)

Project structure
-----------------

```
partx-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── region.py
│   ├── sampler.py
│   ├── partx.py
│   ├── estimator.py
│   └── visualization.py
├── examples/
│   └── run_partx_demo.py
├── figures/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_region.py
```

Enjoy exploring Part-X ideas!
