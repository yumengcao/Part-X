# partx_params.py
# Region format for Part_X: list of (low, high) per dimension
dim = 2
region = [(-5, 5)]*dim#[(-2,2)]*dim#[(-5, 10),(0, 15)]#[(-32.768, 32.768)]*2  # example 2D
method = 'BO'                  # or 'uniform'
budget = 1000000

initial_budget_per_region = round(10*dim)
max_iterations = 10
delta = 0
# Part_X-specific defaults you might want to override:
#initial_budget_per_region = 
#max_iterati
# (Optional) any other parameters your Part_X expects; keep names consistent.