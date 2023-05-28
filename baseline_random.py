from batched_env import BlackBox
from sklearn_GP import GP as sklearnGP
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import torch

def run(dims):
    env = BlackBox(batch_size=2, dims = dims, use_GP = False)
    env.reset()

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = env.action_space.sample()
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, done, info = env.step(torch.tensor(act).to(torch.device("cuda")), isindex=False)
        done = done[0]



    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak


def run(n, dims, learning_rate = None, training_iters = None, approximate = None, noise = None, batch_size = 512):
    """use_all: Whether to use all training points (the full 50, which includes duplicates) or just without duplicates"""
    #NOTE: Shady?
    env = BlackBox(batch_size=2, dims = dims, use_GP = False)
    env.reset()

    rewards = []
    lengths = []
    peaks = []

    while len(peaks) < n:
        act = env.action_space.sample()
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, dones, info = env.step(torch.tensor(act).to(torch.device("cuda")), isindex=False)
        if torch.sum(dones) > 0:
            rewards += info["episodic_returns"][dones].tolist()
            lengths += info["episodic_length"][dones].tolist()
            peaks += info["peak"][dones].tolist()

    return rewards, lengths, peaks
