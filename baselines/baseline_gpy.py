from tqdm import tqdm
import torch
#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
def run(n, config, learning_rate = None, training_iters = None, approximate = None, noise = None, kernel = None):
    """use_all: Whether to use all training points (the full 50, which includes duplicates) or just without duplicates"""
    #NOTE: Shady?
    if learning_rate is not None:
        config.GP_learning_rate = learning_rate
    if training_iters is not None:
        config.GP_training_iters = training_iters
    if approximate is not None:
        config.approximate = approximate
    if noise is not None:
        config.noise = noise
    if kernel is not None:
        config.kernels = [kernel]
    env = BlackBox(config=config)
    env.reset()

    rewards = []
    lengths = []
    peaks = []


    with tqdm(total=n, desc = "Baselining GPY", leave = False) as pbar:
        while len(peaks) < n:
            act = env.GP.get_next_point(return_idx = True)
            _, _, dones, info = env.step(act)
            if torch.sum(dones) > 0:
                rewards += info["episodic_returns"][dones].tolist()
                lengths += info["episodic_length"][dones].tolist()
                peaks += info["peak"][dones].tolist()
                pbar.update(torch.sum(dones).item())

    return rewards, lengths, peaks
