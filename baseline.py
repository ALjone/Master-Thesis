from baselines.baseline_random import run as run_random
from baselines.baseline_gpy import run as run_gpy
from agent_performance import baseline as agent_baseline
import warnings
from tqdm import tqdm
import numpy as np
from utils import load_config, pretty_print_results
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
        
    pretty_print_results(rewards, lengths, peaks)

def baseline(training, run, n, dims = 2, batch_size = 512):
    config = load_config("configs\\training_config.yml" if training else "configs\\testing_config.yml")
    config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    rewards, lengths, peaks = run(n, config)
    pretty_print_results(rewards, lengths, peaks)


n = 100000
dims = 2
batch_size = 2048
time_model = "models\\Goldstein-Price.t"
no_time_model = time_model[:-2] + " no time.t"

print("Baselining training env\n")

#WITH TIME:
print(f"\tBaseline random with n = {n}:")
baseline(True, run_random, n, dims, batch_size=batch_size)
print(f"\n\tBaseline gpy with n = {n}:")
baseline(True, run_gpy, n, dims, batch_size=batch_size)
print(f"\n\tTime agent with n = {n}:")
agent_baseline(True, n, time_model, True, dims, batch_size=512)
print(f"\n\tNo time agent with n = {n}:")
agent_baseline(True, n, no_time_model, False, dims, batch_size=512)

exit()
print("\n\nBaselining test env\n")

#WITH TIME:
print(f"\tBaseline random with n = {n}:")
baseline(False, run_random, n, dims)
print(f"\n\tBaseline gpy with n = {n}:")
baseline(False, run_gpy, n, dims)
#print(f"\nBaseline sklearn with n = {n}:")
#baseline_sklearn(False, run_sklearn, n, dims)
