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

lrs = [0.5, 0.1, 0.01]
training_iters = [100, 50, 25]
use_alls = [True, False]
lr_dict = {lr: [] for lr in lrs}
iter_dict = {iters: [] for iters in training_iters}
use_all_dict = {use_all: [] for use_all in use_alls}

#Too low? Okay I s'pose
n = 300
T = 100

resolution = 30
domain = (-1, 1)
env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, T = T, kernels=[RBFKernel, MaternKernel, RQKernel])

for lr in lrs:
    for iters in training_iters:
        for use_all in use_alls:
            print(f"Learning rate: {lr}     Iterations: {iters}     Use all: {use_all}")
            reward, length, peak = baseline_gpy(T, n, env, None, learning_rate=lr, training_iters=iters, use_all = use_all)
            print("\n")
            if peak > best_peak:
                best_length = length
                best_reward = reward
                best_peak = peak
                best_params = {"lr":lr, "iters": iters, "use_all": use_all}
            lr_dict[lr].append(peak)
            iter_dict[iters].append(peak)
            use_all_dict[use_all].append(peak)
            
print("\n")
print("Averages:\n")
print("Leaning rate:")
for lr, peaks in lr_dict.items():
    print(f"\tLearning rate: {lr}   Average: {round(np.mean(peaks), 4)}")

print("Training iterations:")
for iters, peaks in iter_dict.items():
    print(f"\tTraining iterations: {iters}   Average: {round(np.mean(peaks), 4)}")

print("Use all:")
for use_all, peaks in use_all_dict.items():
    print(f"\tUse all: {use_all}   Average: {round(np.mean(peaks), 4)}")


print("\n\n")

print("Best params:")
print(f"\tLearning rate: {best_params['lr']}     Iterations: {best_params['iters']}     Use all: {best_params['use_all']}")
print(f"\tReward: {round(best_reward, 4)}       Peak: {round(best_peak, 4)}     Length: {best_length}")