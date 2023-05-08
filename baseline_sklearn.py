from batched_env import BlackBox
from Old.GP import GP as sklearnGP
from GPY import GP as gpyGP
from sklearn.gaussian_process.kernels import RBF
import torch
from scipy.stats import norm
from tqdm import tqdm
def make_action(action):
    return torch.stack((torch.tensor(action), torch.tensor([0.12, 0.31])), dim = 0).to(torch.device("cuda"))



def EI(u, std, biggest, e = 0.01):
    if std <= 0:
        print("std under 0")
        return 0
    Z = (u-biggest-e)/std
    return (u-biggest-e)*norm.cdf(Z)+std*norm.pdf(Z)

def run(T, max_length = None):
    resolution = 30
    domain = (-1, 1)
    env = BlackBox(resolution, domain = domain, batch_size=2, num_init_points=2, dims = 2, use_GP = False, T = T)
    env.reset()
    checked_points = env.actions_for_gp[0, :2]
    value_points = env.values_for_gp[0, :2]
    
    sklearn = sklearnGP([RBF()*1], EI, (domain, domain), resolution, ("One", "Two"), checked_points = checked_points.cpu().numpy(), values_found=value_points.cpu().numpy())

    #TODO: Seems to still be a bug related to scaling
    done = False
    while not done:
        act = sklearn.get_next_point()
        next = (make_action(act)-(resolution//2))/(resolution//2)
        #print("x:", next[0][0].item(), "y:", next[0][1].item())
        _, _, done, info = env.step(next, transform=False)
        done = done[0]
        sklearn.update_points(env.actions_for_gp[0, env.batch_step[0]-1].cpu().numpy(), env.values_for_gp[0, env.batch_step[0]-1].cpu().numpy())
        #print("Adding points:", env.actions_for_gp[0, env.batch_step[0]-1].cpu().numpy())
        #print("Checked points:")
        #print(sklearn.checked_points)
        #sklearn.render()
        #print()
        if max_length is not None and env.batch_step[0] > max_length:
            break

    r = info["episodic_returns"][0]
    length = info["episodic_length"][0]
    peak = info["peak"][0]

    return r, length, peak


def baseline_sklearn(T, n, max_length = None):
    reward = 0
    length = 0
    peak = 0
    for _ in tqdm(range(n), disable=False, desc="Baselining sklearn", leave = False):
        r, l, p = run(T, max_length = max_length)
        reward += r
        length += l
        peak += p

    print("\tReward:", round((reward/n).item(), 4), "Length:", round((length/n).item()-2, 4), "Peak:", round((peak/n).item(), 4))