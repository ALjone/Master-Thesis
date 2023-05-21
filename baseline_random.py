from batched_env import BlackBox
from sklearn_GP import GP as sklearnGP
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import torch

def run(max_length, dims):
    env = BlackBox(batch_size=2, dims = dims, use_GP = False)
    env.reset()

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = env.action_space.sample()
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, done, info = env.step(torch.tensor(act).to(torch.device("cuda")), transform=False)
        done = done[0]
        if max_length is not None and env.batch_step[0] > max_length:
            break

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak
