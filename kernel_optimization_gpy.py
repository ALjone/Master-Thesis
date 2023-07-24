import numpy as np
from baselines.baseline_gpy import run
import warnings
from tqdm import tqdm
from utils import load_config
import gpytorch
import torch
warnings.filterwarnings("ignore")


def baseline(n, kernel_classes, operation, dims, batch_size):
    config = load_config("configs\\training_config.yml")
    config.dims = dims
    config.batch_size = batch_size
    config.verbose = 0
    config.kernel_classes = kernel_classes
    config.operation = operation
    rewards, lengths, peaks = run(n, config)
    n = len(peaks)
    reward_avg = sum(rewards)/n
    length_avg = sum(lengths)/n
    peak_avg = sum(peaks)/n

    reward_std = round(np.std(rewards)/np.sqrt(n), 4)
    length_std = round(np.std(lengths)/np.sqrt(n), 4)
    peak_std = round(np.std(peaks)/np.sqrt(n), 4)
    
    print(f"\tReward: {round(reward_avg, 4)} ± {reward_std}, Length: {round(length_avg, 4)} ± {length_std}, Peak: {round(peak_avg, 4)} ± {peak_std}")

    return reward_avg, length_avg, peak_avg


best_peak = -np.inf
best_length = None
best_reward = None
best_params = None

import itertools

def get_kernel_combinations(kernel_classes, combination_size):
    combinations = []
    for combination_size in range(1, combination_size+1):
        for combo in itertools.combinations(kernel_classes, combination_size):
            # Additive combination
            combinations.append(('add', combo))
            # Multiplicative combination
            combinations.append(('mul', combo))
    return combinations

# Define kernel classes
kernel_classes = [gpytorch.kernels.RBFKernel, gpytorch.kernels.MaternKernel, gpytorch.kernels.PeriodicKernel, gpytorch.kernels.RQKernel, gpytorch.kernels.LinearKernel]

#Too low? Okay I s'pose
n = 10000
dims = 2
batch_size = 2048*2
combination_size = 2



# Get combinations of size 2
combinations = get_kernel_combinations(kernel_classes, combination_size)
print("Running with", len(combinations), "combinations")
# print the combinations
for operation, combo in combinations:
    print(f'Operation: {operation}, Kernels: {[k.__name__ for k in combo]}')


results = []

# New dictionary to store the sum of peak values and count for each kernel
peak_dict = {kernel_class.__name__: [0, 0] for kernel_class in kernel_classes}

# Initialize dictionary to store sum of peak values and count for each operation
operation_dict = {'add': [0, 0], 'mul': [0, 0]}

for i, (operation, kernel_classes) in enumerate(combinations):
    kernels = [kernel_class() for kernel_class in kernel_classes]
    print(f"{i+1}/{len(combinations)}  Kernels:", [k.__name__ for k in kernel_classes], "Operation:", operation)
    try:
        reward, length, peak = baseline(n, kernel_classes, operation, dims, batch_size)
    except:
        print("Unstable kernel\n")
        continue
    print("\n")
    
    # Save the results
    results.append((reward, length, peak, operation, [k.__name__ for k in kernel_classes]))

    if peak > best_peak:
        best_length = length
        best_reward = reward
        best_peak = peak
        best_params = kernel_classes
    
    # Add the peak value to each kernel's total in the dictionary, and increment the count
    for kernel_class in kernel_classes:
        peak_dict[kernel_class.__name__][0] += peak
        peak_dict[kernel_class.__name__][1] += 1

    # Similarly, add the peak value to the operation's total in operation_dict, and increment the count
    operation_dict[operation][0] += peak
    operation_dict[operation][1] += 1

# Compute the average peak for each kernel type and operation
average_peaks = {kernel_name: peak_total / count for kernel_name, (peak_total, count) in peak_dict.items()}
average_operation_peaks = {operation: peak_total / count for operation, (peak_total, count) in operation_dict.items()}

# Sort the results based on peak
sorted_results = sorted(results, key=lambda x: x[2], reverse=True)


print("\nAll Combinations (sorted by peak):")
for reward, length, peak, operation, kernel_names in sorted_results:
    print(f"Operation: {operation}, Kernels: {kernel_names}")
    print(f"\tReward: {round(reward, 4)}       Length: {round(length, 4)}     Peak: {round(peak, 4)}\n")

print("\nAverage peak per kernel type:")
for kernel_name, avg_peak in average_peaks.items():
    print(f"Kernel: {kernel_name}, Average Peak: {round(avg_peak, 4)}")

print("\nAverage peak per operation:")
for operation, avg_peak in average_operation_peaks.items():
    print(f"Operation: {operation}, Average Peak: {round(avg_peak, 4)}")
