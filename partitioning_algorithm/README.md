# Partitioning Algorithm for Iterative Subregion Decomposition

This module implements a tree-based partitioning algorithm for recursive subregion splitting across multiple iterations. It is designed for applications like Gaussian Process modeling, region classification, and adaptive sampling.

## Key Features

- **Hierarchical Tree Structure**: Each region is tracked across iterations using a tree-like dictionary structure.
- **Flexible Splitting**: Regions can be split into multiple subregions along specified dimensions.
- **Sample Allocation Propagation**: Parent node sample allocations are proportionally divided among child nodes.
- **Preserves Unsplit Regions**: Regions that are not split retain their ID and are carried over to the next iteration.

---

## Tree Structure

The `tree` variable stores all region bounds and relationships:

```python
tree = {
    'iter_0': {
        'parent_{}': {
            'r1_L0': [(low1, high1), (low2, high2)]
        }
    },
    'iter_1': {
        'parent_r1_L0': {
            'r1_L1': [...],
            'r2_L1': [...]
        }
    },
    ...
}
```

- **Key Format**: `r{j}_L{level}`
- **Parent Key**: always `parent_<parent_id>`, where `parent_id` comes from previous iteration
- Regions that are not partitioned are kept as-is under their own parent key.

---

## Partitioning Class Usage

```python
from partition import Partitioning

partitioner = Partitioning(
    tree=tree,
    sample_allocation=sample_allocation,
    dim_index=dim_index,
    part_number=part_number,
    dim=2,
    current_iter=1
)

tree_out, child_alloc_out, dim_index_out = partitioner.partition()
```

---

## Input Descriptions

- `tree`: The current region hierarchy (as shown above)
- `sample_allocation`: Dict of sample budgets per region in current iteration
- `dim_index`: Dict indicating which dimension to split for each region
- `part_number`: Number of splits per region
- `dim`: Dimension of the space
- `current_iter`: The index of current iteration (starting from 0)

---

## Output

- `tree_out`: Updated tree including the new iteration
- `child_alloc_out`: Sample allocations for new subregions
- `dim_index_out`: Dimension indices for next split (cyclic or custom)

---

## Example

```python
tree = {
    'iter_0': {'parent_{}': {'r1_L0': [(0, 10), (0, 10)]}}
}
sample_allocation = {'r1_L0': 30}
dim_index = {'r1_L0': 0}
part_number = {'r1_L0': 3}

partitioner = Partitioning(tree, sample_allocation, dim_index, part_number, dim=2, current_iter=0)
tree_out, alloc_out, dim_out = partitioner.partition()
```

---

## Notes

- Levels start at `L0`, region IDs at `r1`
- Subregions that are not split retain their ID and parent mapping
- This format simplifies visualizing and analyzing the evolution of regions over time