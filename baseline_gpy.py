from batched_env import BlackBox
from GPY import GP as gpyGP
from gpytorch.kernels import RBFKernel, MaternKernel, CosineKernel, PolynomialKernel, LinearKernel
import torch
from scipy.stats import norm
from tqdm import tqdm
def make_action(action, dims):
    return torch.stack((torch.tensor(action), torch.tensor([0.12 for _ in range(dims)])), dim = 0).to(torch.device("cuda"))

def EI(u, std, biggest, e = 0.01):
    if std <= 0:
        print("std under 0")
        return 0
    Z = (u-biggest-e)/std
    return (u-biggest-e)*norm.cdf(Z)+std*norm.pdf(Z)

def run(max_length, dims, learning_rate = 0.01, training_iters = 100):
    """use_all: Whether to use all training points (the full 50, which includes duplicates) or just without duplicates"""
    env = BlackBox(batch_size=2, dims = dims)
    resolution = env.resolution
    env.GP.learning_rate = learning_rate
    env.GP.training_iters = training_iters
    env.reset()

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = env.GP.get_next_point(EI, torch.max(torch.stack(env.values_for_gp[0])).cpu().numpy())
        next = (make_action(act, dims)-(resolution//2))/(resolution//2) #TODO: Why?
        _, _, done, info = env.step(next, transform=False)
        done = done[0]

        if max_length is not None and env.batch_step[0] > max_length:
            break

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak
