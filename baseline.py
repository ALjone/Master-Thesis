from baselines.baseline_random import run as run_random
from baselines.baseline_sklearn import run as run_sklearn
from baselines.baseline_gpy import run as run_gpy
import warnings
from tqdm import tqdm
import numpy as np
from utils import load_config
warnings.filterwarnings("ignore")

def baseline_sklearn(training, run, n, dims = 2, batch_size = 512):
    config = load_config("configs\\training_config.yml" if training else "configs\\testing_config.yml")
    config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    rewards = []
    lengths = []
    peaks = []
    for _ in tqdm(range(n), disable=False, desc="Baselining", leave=False):
        r, l, p = run(config)
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

def baseline(training, run, n, dims = 2, batch_size = 512):
    config = load_config("configs\\training_config.yml" if training else "configs\\testing_config.yml")
    config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    rewards, lengths, peaks = run(n, config)
    n = len(peaks)
    reward_avg = round(sum(rewards)/n, 4)
    length_avg = round(sum(lengths)/n, 4)
    peak_avg = round(sum(peaks)/n, 4)

    reward_std = round(np.std(rewards)/np.sqrt(n), 4)
    length_std = round(np.std(lengths)/np.sqrt(n), 4)
    peak_std = round(np.std(peaks)/np.sqrt(n), 4)
    
    print(f"\t\tReward: {reward_avg} ± {reward_std}, Length: {length_avg} ± {length_std}, Peak: {peak_avg} ± {peak_std}")


n = 5000
dims = 2
print("Baselining training env\n")

#WITH TIME:
print(f"\tBaseline random with n = {n}:")
baseline(True, run_random, n, dims)
print(f"\n\tBaseline gpy with n = {n}:")
baseline(True, run_gpy, n, dims)
#print(f"\nBaseline sklearn with n = {n}:")
#baseline_sklearn(True, run_sklearn, n, dims)


print("\n\nBaselining test env\n")

#WITH TIME:
print(f"\tBaseline random with n = {n}:")
baseline(False, run_random, n, dims)
print(f"\n\tBaseline gpy with n = {n}:")
baseline(False, run_gpy, n, dims)
#print(f"\nBaseline sklearn with n = {n}:")
#baseline_sklearn(False, run_sklearn, n, dims)
