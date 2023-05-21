from batched_env import BlackBox

import torch
from scipy.stats import norm
from tqdm import tqdm
from acquisition_functions import EI
from utils import make_action

def run(max_length, dims, learning_rate = None, training_iters = None, approximate = None, noise = None):
    """use_all: Whether to use all training points (the full 50, which includes duplicates) or just without duplicates"""
    #NOTE: Shady?
    params = {"batch_size": 2, "dims": dims}
    if learning_rate is not None:
        params["GP_learning_rate"] = learning_rate
    if training_iters is not None:
        params["GP_training_iters"] = training_iters
    if approximate is not None:
        params["approximate"] = approximate
    if noise is not None:
        params["noise"] = noise

    env = BlackBox(**params)
    resolution = env.resolution
    env.reset()

    done = False
    while not done:
        act = env.GP.get_next_point(torch.max(torch.stack(env.values_for_gp[0])).cpu().numpy())
        next = (act-(resolution//2))/(resolution//2) #TODO: Why?
        _, _, done, info = env.step(next, transform=False)
        done = done[0]

        if max_length is not None and env.batch_step[0] > max_length:
            break

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak
