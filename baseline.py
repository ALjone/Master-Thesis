from baseline_random import run as run_random
from baseline_sklearn import run as run_sklearn
from baseline_gpy import run as run_gpy
import warnings
from tqdm import tqdm
import numpy as np
warnings.filterwarnings("ignore")

def baseline_sklearn(run, n, dims = 2):
    rewards = []
    lengths = []
    peaks = []
    for _ in tqdm(range(n), disable=False, desc="Baselining", leave=False):
        r, l, p = run(dims)
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

def baseline(run, n, dims = 2, batch_size = 512):
    rewards, lengths, peaks = run(n, dims, batch_size = batch_size)
    n = len(peaks)
    reward_avg = round(sum(rewards)/n, 4)
    length_avg = round(sum(lengths)/n, 4)
    peak_avg = round(sum(peaks)/n, 4)

    reward_std = round(np.std(rewards)/np.sqrt(n), 4)
    length_std = round(np.std(lengths)/np.sqrt(n), 4)
    peak_std = round(np.std(peaks)/np.sqrt(n), 4)
    
    print(f"Reward: {reward_avg} ± {reward_std}, Length: {length_avg} ± {length_std}, Peak: {peak_avg} ± {peak_std}")


n = 5000
dims = 3
print("\n\n")

#WITH TIME:
print(f"Baseline random with n = {n}:")
baseline(run_random, n, dims)
print(f"\nBaseline gpy with n = {n}:")
baseline(run_gpy, n, dims)
print(f"\nBaseline sklearn with n = {n}:")
baseline_sklearn(run_sklearn, n, dims)
