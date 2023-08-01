#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
from tqdm import tqdm
import torch


def run(n, config):
    #NOTE: Shady?
    config.use_GP = False
    env = BlackBox(config)
    env.reset()

    rewards = []
    lengths = []
    peaks = []

    while len(peaks) < n:
        act = env.action_space.sample()
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, dones, info = env.step(torch.tensor(act).to(torch.device("cpu")))
        if torch.sum(dones) > 0:
            rewards += info["episodic_returns"][dones].tolist()
            lengths += info["episodic_length"][dones].tolist()
            peaks += info["peak"][dones].tolist()

    return rewards, lengths, peaks
