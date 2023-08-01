from tqdm import tqdm
import torch        
import numpy as np
#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
#from agents.pix_2_pix_agent import Agent
#from agents.ei_argmax_weighting import Agent
from agents.pointwise import Agent
from utils import load_config
from matplotlib import pyplot as plt
import matplotlib as mpl

config = load_config("configs\\training_config.yml")

env = BlackBox(config)
agent: Agent = Agent(env.observation_space, config.layer_size, env.dims).to(torch.device("cpu"))
#agent.load_state_dict(torch.load("model_with_positional_encoding.t"))
#agent.load_state_dict(torch.load("models/With positional encoding, temperature, convex, relu, high res.t"))
agent.load_state_dict(torch.load("models/multimodal 50 64.t"))


R = 0
t = 0
sims = 100

assert config.dims == 2, "Only 2d supported"

x_actions = []
y_actions = []

avg = np.zeros(tuple(config.resolution for _ in range(config.dims)))

func_avg = np.zeros(tuple(config.resolution for _ in range(config.dims)))

for i in tqdm(range(sims)):
    s = env.reset()
    #func_avg += torch.mean(env.func_grid, dim = 0).cpu().numpy()
    func_avg += torch.mean(env.GP.mean, dim = 0).cpu().numpy()
    continue
    dones = [False]
    #continue
    while not dones[0]:
        action, _, _, _ = agent.get_action_and_value(s)
        logits, probs = agent.get_2d_logits_and_prob(s)
        x_actions += list(action[0].cpu())
        y_actions += list(action[1].cpu())

        s, r, dones, info = env.step(action)
        R += r
        t += 1
        avg += np.mean(logits, axis = 0)

avg /=sims

func_avg /=sims

im = plt.imshow(func_avg)
plt.colorbar(im)
plt.gca().invert_yaxis()
plt.title("Avg of func grids checked during test")
plt.show()

im = plt.imshow(avg)
plt.colorbar(im)
plt.gca().invert_yaxis()
plt.title("Avg of all logits checked during test")
plt.show()

resolution = config.resolution
fig, ax = plt.subplots()
h = ax.hist2d(x_actions, y_actions, bins = [np.arange(0, resolution-1, resolution), np.arange(0, resolution-1, resolution)])
fig.colorbar(h[3], ax=ax)
plt.title("Heat map of actions taken")
plt.show()


fig, ax = plt.subplots()
h = ax.hist2d(x_actions, y_actions, bins = [np.arange(0, resolution-1, resolution), np.arange(0, resolution-1, resolution)], norm=mpl.colors.LogNorm())
fig.colorbar(h[3], ax=ax)
plt.title("Log normalized heat map of actions taken")
plt.show()


print(f"Avg reward {round(R.cpu().numpy()/sims, 3)}, Avg steps {round(t/sims, 3)}")