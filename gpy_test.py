import numpy as np
from baseline_gpy import baseline_gpy
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel
import warnings

from batched_env import BlackBox
warnings.filterwarnings("ignore")

best_peak = -np.inf
best_length = None
best_reward = None
best_params = None


#Too low? Okay I s'pose
n = 100
T = 100

resolution = 30
domain = (-1, 1)
env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, T = T, kernels=[RBFKernel])

for approximate in [True, False]:
    print("Approximate:", approximate)
    env.GP.approximate = approximate
    reward, length, peak = baseline_gpy(T, n, env, None)
    print("\n")
    if peak > best_peak:
        best_length = length
        best_reward = reward
        best_peak = peak

print("\n\n")

print("Best params:")
print(f"\tLearning rate: {best_params['lr']}     Iterations: {best_params['iters']}     Use all: {best_params['use_all']}")
print(f"\tReward: {round(best_reward, 4)}       Peak: {round(best_peak, 4)}     Length: {best_length}")