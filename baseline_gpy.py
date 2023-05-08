from batched_env import BlackBox
from GPY import GP as gpyGP
from gpytorch.kernels import RBFKernel, MaternKernel, CosineKernel, PolynomialKernel, LinearKernel
import torch
from scipy.stats import norm
from tqdm import tqdm
def make_action(action):
    return torch.stack((torch.tensor(action), torch.tensor([0.12, 0.31])), dim = 0).to(torch.device("cpu"))



def EI(u, std, biggest, e = 0.01):
    if std <= 0:
        print("std under 0")
        return 0
    Z = (u-biggest-e)/std
    return (u-biggest-e)*norm.cdf(Z)+std*norm.pdf(Z)

def run(T, max_length = None):
    resolution = 30
    domain = (-1, 1)
    env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, T = T)
    env.reset()

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = env.GP.get_next_point(EI, torch.max(env.values_for_gp[0]).cpu().numpy())
        next = (make_action(act)-(resolution//2))/(resolution//2)
        _, _, done, info = env.step(next, transform=False)
        done = done[0]

        if max_length is not None and env.batch_step[0] > max_length:
            break

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak


def baseline_gpy(T, n, max_length = None):
    reward = 0
    length = 0
    peak = 0
    for _ in tqdm(range(n), disable=False, desc="Baselining gpy", leave = False):
        r, l, p = run(T, max_length = max_length)
        reward += r
        length += l
        peak += p

    print("\tReward:", round((reward/n).item(), 4), "Length:", round((length/n).item()-2, 4), "Peak:", round((peak/n).item(), 4))