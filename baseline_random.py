from batched_env import BlackBox
from Old.GP import GP as sklearnGP
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import torch

def run(T, max_length = None):
    resolution = 30
    domain = (-1, 1)
    env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, use_GP = False, T = T)
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


def baseline_random(T, n, max_length = None):
    reward = 0
    length = 0
    peak = 0
    for _ in tqdm(range(n), disable=False, desc="Baselining random", leave=False):
        r, l, p = run(T, max_length = max_length)
        reward += r
        length += l
        peak += p

    print("\tReward:", round((reward/n).item(), 4), "Length:", round((length/n).item()-2, 4), "Peak:", round((peak/n).item(), 4))

#find_average_for_start_points(env)