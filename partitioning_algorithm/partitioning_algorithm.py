import copy
from typing import Dict, List, Tuple

class Partitioning:
    def __init__(
        self,
        tree: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]],
        sample_allocation: Dict[str, int],
        dim_index: Dict[str, int],
        part_number: Dict[str, int],
        dim: int,
        region_counter: int,
        allowed_regions: List[str],
        min_samples_per_child: int = 1
    ):
        self.tree = tree
        self.sample_allocation = sample_allocation
        self.dim_index = dim_index
        self.part_number = part_number
        self.dim = dim
        self.region_counter = region_counter
        self.allowed_regions = set(allowed_regions)
        self.min_samples_per_child = int(min_samples_per_child)

    def _validate_bounds(self, region_bounds: List[Tuple[float, float]]):
        assert len(region_bounds) == self.dim, "Region must have bounds for each dimension"
        for i, (low, high) in enumerate(region_bounds):
            if not low < high:
                raise ValueError(f"Invalid bounds in dim {i}: low={low}, high={high}")

    def _split_interval(self, low: float, high: float, num_parts: int) -> List[Tuple[float, float]]:
        width = (high - low) / num_parts
        return [(low + i * width, low + (i + 1) * width) for i in range(num_parts)]

    def _allocate_child_samples(self, total: int, n_children: int) -> List[int]:
        if n_children <= 0:
            return []
        base = total // n_children
        rem = total % n_children
        alloc = [base] * n_children
        for i in range(rem):
            alloc[i] += 1
        return alloc

    def _can_partition(self, total_samples: int, requested_parts: int) -> bool:
        """
        Check whether the region has enough samples to justify partitioning.
        """
        if requested_parts < 2:
            return False
        return total_samples >= requested_parts * self.min_samples_per_child

    def partition(self) -> Tuple[
        Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]],
        Dict[str, int],
        Dict[str, int],
        int
    ]:
        current_iter = max(int(key.split("_")[-1]) for key in self.tree.keys())
        next_iter = f"iter_{current_iter + 1}"
        self.tree[next_iter] = {}

        next_sample_allocation = {}
        next_dim_index = {}

        for parent_key, region_dict in self.tree[f"iter_{current_iter}"].items():
            for region_id, bounds in region_dict.items():
                if region_id not in self.allowed_regions:
                    continue

                self._validate_bounds(bounds)

                requested_parts = self.part_number.get(region_id, 1)
                dim_to_split = self.dim_index.get(region_id, 0)
                total_samples = self.sample_allocation.get(region_id, 0)

                # decide actual partition number
                max_possible_parts = total_samples // self.min_samples_per_child
                actual_parts = min(requested_parts, max_possible_parts)

                # if cannot split into at least 2 parts, keep region unchanged
                if actual_parts < 2 or not self._can_partition(total_samples, actual_parts):
                    if f"parent_{region_id}" not in self.tree[next_iter]:
                        self.tree[next_iter][f"parent_{region_id}"] = {}
                    self.tree[next_iter][f"parent_{region_id}"][region_id] = bounds
                    next_sample_allocation[region_id] = total_samples
                    next_dim_index[region_id] = dim_to_split
                    continue

                low, high = bounds[dim_to_split]
                intervals = self._split_interval(low, high, actual_parts)
                sample_alloc = self._allocate_child_samples(total_samples, actual_parts)

                children = {}
                for i, interval in enumerate(intervals):
                    new_bounds = copy.deepcopy(bounds)
                    new_bounds[dim_to_split] = interval

                    new_region_id = f"r{self.region_counter + 1}_L{current_iter + 1}"
                    children[new_region_id] = new_bounds
                    next_sample_allocation[new_region_id] = sample_alloc[i]
                    next_dim_index[new_region_id] = (dim_to_split + 1) % self.dim
                    self.region_counter += 1

                self.tree[next_iter][f"parent_{region_id}"] = children

        return self.tree, next_sample_allocation, next_dim_index, self.region_counter
# class Partitioning:
#     def __init__(self,
#                  tree: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]],
#                  sample_allocation: Dict[str, int],
#                  dim_index: Dict[str, int],
#                  part_number: Dict[str, int],
#                  dim: int,
#                  region_counter: int,
#                  allowed_regions: List[str]): 
#         self.tree = tree
#         self.sample_allocation = sample_allocation
#         self.dim_index = dim_index
#         self.part_number = part_number
#         self.dim = dim
#         self.region_counter = region_counter
#         self.allowed_regions = set(allowed_regions) 

#     def _validate_bounds(self, region_bounds: List[Tuple[float, float]]):
#         assert len(region_bounds) == self.dim, "Region must have bounds for each dimension"
#         for i, (low, high) in enumerate(region_bounds):
#             if not low < high:
#                 raise ValueError(f"Invalid bounds in dim {i}: low={low}, high={high}")

#     def _split_interval(self, low: float, high: float, num_parts: int) -> List[Tuple[float, float]]:
#         width = (high - low) / num_parts
#         return [(low + i * width, low + (i + 1) * width) for i in range(num_parts)]

#     def _allocate_child_samples(self, total: int, n_children: int) -> List[int]:
#         base = total // n_children
#         rem = total % n_children
#         alloc = [base] * n_children
#         for i in range(rem):
#             alloc[i] += 1
#         return alloc

#     def partition(self) -> Tuple[
#         Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]],
#         Dict[str, int],
#         Dict[str, int],
#         int]:

#         current_iter = max([int(key.split("_")[-1]) for key in self.tree.keys()])
#         next_iter = f"iter_{current_iter + 1}"
#         self.tree[next_iter] = {}

#         next_sample_allocation = {}
#         next_dim_index = {}

#         for parent_key, region_dict in self.tree[f"iter_{current_iter}"].items():
#             for region_id, bounds in region_dict.items():
                
#                 if region_id not in self.allowed_regions:
#                     continue

#                 self._validate_bounds(bounds)

#                 num_parts = self.part_number.get(region_id, 1)
#                 dim_to_split = self.dim_index.get(region_id, 0)
#                 total_samples = self.sample_allocation.get(region_id, 0)

#                 if num_parts == 1:
#                     if f"parent_{region_id}" not in self.tree[next_iter]:
#                         self.tree[next_iter][f"parent_{region_id}"] = {}
#                     self.tree[next_iter][f"parent_{region_id}"][region_id] = bounds
#                     next_sample_allocation[region_id] = total_samples
#                     next_dim_index[region_id] = dim_to_split
#                     continue

#                 low, high = bounds[dim_to_split]
#                 intervals = self._split_interval(low, high, num_parts)
#                 sample_alloc = self._allocate_child_samples(total_samples, num_parts)

#                 children = {}
#                 for i, interval in enumerate(intervals):
#                     new_bounds = copy.deepcopy(bounds)
#                     new_bounds[dim_to_split] = interval
#                     new_region_id = f"r{self.region_counter + 1}_L{current_iter + 1}"
#                     children[new_region_id] = new_bounds
#                     next_sample_allocation[new_region_id] = sample_alloc[i]
#                     next_dim_index[new_region_id] = (dim_to_split + 1) % self.dim
#                     self.region_counter += 1

#                 self.tree[next_iter][f"parent_{region_id}"] = children

#         return self.tree, next_sample_allocation, next_dim_index, self.region_counter