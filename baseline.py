from baseline_random import run as run_random
from baseline_sklearn import run as run_sklearn
from baseline_gpy import run as run_gpy
import warnings
from tqdm import tqdm
import numpy as np
warnings.filterwarnings("ignore")

def baseline(run, n, max_length = None, dims = 2):
    rewards = []
    lengths = []
    peaks = []
    for _ in tqdm(range(n), disable=False, desc="Baselining", leave=False):
        r, l, p = run(max_length, dims)
        rewards.append(r.cpu().numpy())
        lengths.append(l.cpu().numpy())
        peaks.append(p.cpu().numpy())
        
    reward_avg = round(sum(rewards)/n, 4)
    length_avg = round(sum(lengths)/n, 4)
    peak_avg = round(sum(peaks)/n, 4)

    reward_std = round(np.std(rewards)/np.sqrt(n), 4)
    length_std = round(np.std(lengths)/np.sqrt(n), 4)
    peak_std = round(np.std(peaks)/np.sqrt(n), 4)

    print(f"Reward: {reward_avg} ± {reward_std}, Length: {length_avg} ± {length_std}, Peak: {peak_avg} ± {peak_std}")


n = 1000
dims = 2
print("\n\n")

#WITH TIME:
max_length = None
print(f"Baseline random with n = {n}:")
baseline(run_random, n, max_length, dims)
print(f"\nBaseline gpy with n = {n}:")
baseline(run_gpy, n, max_length, dims)
print(f"\nBaseline sklearn with n = {n}:")
baseline(run_sklearn, n, max_length, dims)

exit()

#WITH A SET AMOUNT OF ITERATIONS
max_length = 20+1
print(f"\nBaseline random with max_length: {max_length}, n = {n}:")
baseline(run_random, n, max_length, dims)
print(f"\nBaseline gpy with max_length: {max_length}, n = {n}:")
baseline(run_gpy, n, max_length, dims)
print(f"\nBaseline sklearn with max_length: {max_length}, n = {n}:")
baseline(run_sklearn, n, max_length, dims)