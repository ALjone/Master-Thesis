from baseline_utils import benchmark, find_average_for_start_points
from batched_env import BlackBox

env = BlackBox(30, batch_size=64, dims = 2)

str = benchmark(env, 1000)

find_average_for_start_points(env)