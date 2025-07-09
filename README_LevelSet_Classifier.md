
# Level-Set Classification via Partitioning and Bayesian Optimization

This repository contains the implementation of an iterative algorithm for subregion classification using level-set estimation. The method combines:
- Tree-based space partitioning,
- Gaussian Process (GP) modeling,
- Uncertainty quantification,
- Sample reallocation via uniform or Bayesian sampling.

## 🚀 Getting Started

### Dependencies
- Python >= 3.7
- numpy
- matplotlib
- scikit-learn
- scipy

Install using:
```bash
pip install -r requirements.txt
```

### Example Usage

```bash
python part_x.py \
  --region "[(0, 10), (0, 10)]" \
  --method BO \
  --function "lambda x: x[0]**2 + x[1]**2 - 25"
```

## 🧠 Algorithm Overview

### Step-by-Step Breakdown

1. **Partitioning**:
   - The input region is recursively split into subregions using a binary (or multi-way) partitioning tree.
   - The number of partitions per region is determined by a score (e.g., signal-to-noise ratio from GP).

2. **Scoring and GP Modeling**:
   - Each subregion is modeled using a Gaussian Process (GP).
   - Confidence intervals are calculated using Sobol sequences as evaluation points.
   - The score = |mean| / (std + epsilon) is used for subregion prioritization.

3. **Sampling Strategy**:
   - A combination of base-rate and score-inverse sampling is used to allocate budgets per subregion.
   - Sampling is done via:
     - **Uniform**: for exploration
     - **Bayesian Optimization (BO)**: for exploitation using Expected Improvement (EI)

4. **Classification**:
   - Each region is classified into:
     - `θ+` (safe), `θ−` (unsafe), or `θ?` (undefined)
   - This is based on the confidence interval falling above/below/straddling the decision boundary (e.g., `f(x) = 0`)

5. **Tree Tracking**:
   - A hierarchical dictionary `tree = {iter_k: {'parent_r1_Li': {'r2_Li+1': [...], ...}}}` stores subregion splits across iterations.

6. **Visualization**:
   - Colored partitions (`red = θ−`, `green = θ+`, `blue = θ?`) with overlaid sample points.
   - Contour of `f(x) = 0` is plotted for reference.

## 📦 Folder Structure

```
.
├── part_x.py                  # Main entry point
├── Graphing/
│   ├── partition_plot.py      # Visualize regions and samples
│   └── sampling_plot.py       # Handles sample coloring
├── Partition/
│   └── partition_tree.py      # Handles recursive space splitting
├── Model_construction/
│   └── GP_Model.py            # GP model + confidence interval estimation
├── Sampling_Method/
│   └── Bayesian_optimizer.py  # BO sampling logic
├── Utils/
│   └── allocation.py          # Budget reallocation by region scores
```

## 🧪 Testing

```python
# Example function: f(x) = x[0]^2 + x[1]^2 - 25
python part_x.py --region "[(0, 10), (0, 10)]" --method BO --function "lambda x: x[0]**2 + x[1]**2 - 25"
```

## 🧩 Debugging Tips

- If you get:
  - `ValueError: Inconsistent bounds!`: Check if subregion bounds are malformed.
  - `TypeError: float() argument must be ...`: Ensure Y is numeric, not a function.
- For slow convergence or warnings:
  - Use `StandardScaler` on input/output before GP.
  - Limit `N_gp` (e.g., 16–30) to avoid excessive Sobol evaluations.

## 📈 Boxplot for Score Analysis
To compare scoring behavior across iterations (scaled vs unscaled), plot boxplots per iteration:
```python
import matplotlib.pyplot as plt
plt.boxplot(score_list_per_iter)
```

## ✍️ License
This implementation is for research and academic use only.
