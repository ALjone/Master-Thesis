from batched_env import BlackBox
from tqdm import tqdm
import torch

def run(n, dims, learning_rate = None, training_iters = None, approximate = None, noise = None, batch_size = 512):
    """use_all: Whether to use all training points (the full 50, which includes duplicates) or just without duplicates"""
    #NOTE: Shady?
    params = {"batch_size": batch_size, "dims": dims}
    if learning_rate is not None:
        params["GP_learning_rate"] = learning_rate
    if training_iters is not None:
        params["GP_training_iters"] = training_iters
    if approximate is not None:
        params["approximate"] = approximate
    if noise is not None:
        params["noise"] = noise

    env = BlackBox(**params)
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
