from env.batched_env import BlackBox
from sklearn_GP import GP as sklearnGP
from sklearn.gaussian_process.kernels import RBF, Matern
import torch
from acquisition_functions import EI
from utils import make_action


def run(config):
    config.use_GP = False
    env = BlackBox(config)
    env.reset()
    checked_points = torch.stack(env.actions_for_gp[0]).cpu().numpy()
    value_points = torch.stack(env.values_for_gp[0]).cpu().numpy()
    
    sklearn = sklearnGP([RBF()*1, Matern()*1], EI, [(env.x_min, env.x_max) for _ in range(config.dims)], env.resolution, ("One", "Two"), checked_points = checked_points, values_found=value_points)

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = sklearn.get_next_point()
        next = (make_action(act, config.dims))
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, done, info = env.step(next.long())
        done = done[0]
        sklearn.update_points(env.actions_for_gp[0][-1].cpu().numpy(), env.values_for_gp[0][-1].cpu().numpy())

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak
