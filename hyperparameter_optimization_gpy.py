import numpy as np
from baseline_gpy import run
from gpytorch.kernels import RBFKernel, MaternKernel
import warnings
from batched_env import BlackBox
from tqdm import tqdm
warnings.filterwarnings("ignore")


def baseline(n, max_length, dims, learning_rate, iterations, approx, noise):
    rewards = []
    lengths = []
    peaks = []
    for _ in tqdm(range(n), disable=False, desc="Baselining", leave=False):
        r, l, p = run(max_length, dims, learning_rate, iterations, approx, noise)
        rewards.append(r.cpu().numpy())
        lengths.append(l.cpu().numpy())
        peaks.append(p.cpu().numpy())
        
    reward_avg = sum(rewards)/n
    length_avg = sum(lengths)/n
    peak_avg = sum(peaks)/n

    peak_std = round(np.std(peaks)/np.sqrt(n), 4)
    
    print(f"\tPeak: {round(peak_avg, 4)} ± {peak_std}")

    return reward_avg, length_avg, peak_avg




best_peak = -np.inf
best_length = None
best_reward = None
best_params = None

lrs = [0.1, 0.01]
training_iters = [100, 200]
approxs = [True, False]
noises = [None, 0.1, 0.00001]

lr_dict = {lr: [] for lr in lrs}
iter_dict = {iters: [] for iters in training_iters}
approx_dict = {approx: [] for approx in approxs}
noise_dict = {noise: [] for noise in noises}


#Too low? Okay I s'pose
n = 200
dims = 2

for lr in lrs:
    for iters in training_iters:
        for approx in approxs:
            for noise in noises:
                print(f"Learning rate: {lr}     Iterations: {iters}     Approx: {approx}   Noise: {noise}")
                reward, length, peak = baseline(n, None, dims, lr, iters, approx, noise)
                print("\n")
                if peak > best_peak:
                    best_length = length
                    best_reward = reward
                    best_peak = peak
                    best_params = {"lr":lr, "iters": iters, "approx": approx}
                lr_dict[lr].append(peak)
                iter_dict[iters].append(peak)
                approx_dict[approx].append(peak)
                noise_dict[noise].append(peak)
            
print("\n")
print("Averages:\n")
print("Leaning rate:")
for lr, peaks in lr_dict.items():
    print(f"\tLearning rate: {lr}   Average: {round(np.mean(peaks), 4)}")

print("Training iterations:")
for iters, peaks in iter_dict.items():
    print(f"\tTraining iterations: {iters}   Average: {round(np.mean(peaks), 4)}")

print("Approx:")
for approx, peaks in approx_dict.items():
    print(f"\tApprox: {approx}   Average: {round(np.mean(peaks), 4)}")

print("Noise:")
for noise, peaks in noise_dict.items():
    print(f"\tNoise: {noise}   Average: {round(np.mean(peaks), 4)}")


print("\n\n")

print("Best params:")
print(f"\tLearning rate: {best_params['lr']}     Iterations: {best_params['iters']}     Approx: {best_params['approx']}")
print(f"\tReward: {round(best_reward, 4)}       Peak: {round(best_peak, 4)}     Length: {round(best_length, 4)}")