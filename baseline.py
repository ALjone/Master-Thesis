from baseline_random import baseline_random
from baseline_sklearn import baseline_sklearn
from baseline_gpy import baseline_gpy
from gpytorch.kernels import RBFKernel, MaternKernel
from batched_env import BlackBox
import warnings
warnings.filterwarnings("ignore")

resolution = 30
domain = (-1, 1)
T = 100
env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, T = T, kernels=[RBFKernel, MaternKernel])

n = 20
print("\n\n")

#WITH TIME:
T = 100
max_length = None
print(f"Baseline random with time: {T}, n = {n}:")
baseline_random(T, n, max_length)
print(f"\nBaseline gpy with time: {T}, n = {n}:")
baseline_gpy(T, n, env, max_length, learning_rate=0.1, training_iters=100, use_all=True)
print(f"\nBaseline sklearn with time: {T}, n = {n}:")
baseline_sklearn(T, n, max_length)

#WITH A SET AMOUNT OF ITERATIONS
T = 10000000000
max_length = 20
print(f"\nBaseline random with max_length: {max_length}, n = {n}:")
baseline_random(T, n, max_length)
print(f"\nBaseline gpy with max_length: {max_length}, n = {n}:")
baseline_gpy(T, n, env, max_length, learning_rate=0.1, training_iters=100, use_all=True)
print(f"\nBaseline sklearn with max_length: {max_length}, n = {n}:")
baseline_sklearn(T, n, max_length)