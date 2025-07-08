# test_partition.py

from partitioning_algorithm import Partitioning

# ----------- Initial Setup -----------
tree_0 = {
    'iter_0': {
        'parent_r0_L0': {
            'r0_L0': [(0, 10), (0, 10)]
        }
    }
}

sample_allocation_0 = {
    'r0_L0': 100
}

dim_index_0 = {
    'r0_L0': 0
}

part_number_0 = {
    'r0_L0': 2
}

# ----------- Run First Partitioning -----------
partitioner_0 = Partitioning(tree_0, sample_allocation_0, dim_index_0, part_number_0, dim=2, region_counter=0)
tree_1, alloc_1, dim_index_1, counter_1 = partitioner_0.partition()

print("--- Iteration 1 ---")
print("Updated Tree:", tree_1)
print("Sample Allocation:", alloc_1)
print("Next Dim Index:", dim_index_1)
print("Region Counter:", counter_1)

# ----------- Run Second Partitioning -----------
part_number_1 = {
    'r1_L1': 2,
    'r2_L1': 2
}

partitioner_1 = Partitioning(tree_1, alloc_1, dim_index_1, part_number_1, dim=2, region_counter=counter_1)
tree_2, alloc_2, dim_index_2, counter_2 = partitioner_1.partition()

print("\n--- Iteration 2 ---")
print("Updated Tree:", tree_2)
print("Sample Allocation:", alloc_2)
print("Next Dim Index:", dim_index_2)
print("Region Counter:", counter_2)
