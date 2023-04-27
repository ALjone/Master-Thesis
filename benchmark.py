from benchmark_utils import benchmark
from batched_env import BlackBox

env = BlackBox(30, batch_size=64, dims = 2)

str = benchmark(env, 100)