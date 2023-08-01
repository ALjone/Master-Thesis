from matplotlib import pyplot as plt
import torch        
import numpy as np
#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
#from agents.pix_2_pix_agent import Agent
#from agents.ei_argmax_weighting import Agent
from agents.pointwise import Agent
from utils import load_config

config = load_config("configs\\training_config.yml")
config.use_time = True
env = BlackBox(config)
agent: Agent = Agent(env.observation_space, config.layer_size, env.dims)
agent.load_state_dict(torch.load("models/multimodal temperature.t"))

s = env.reset()

#Mean, std, time in point, time spent, best pred, max time
point = [0.5, 1, 0, 0, 0.7, 0.1]
AFs = np.zeros((100, 100))
critic_vals = np.zeros((100, 100))
for i, time in enumerate(np.linspace(0, 1, 100)):
    for j, mean in enumerate(np.linspace(0, 1, 100)):
        point[4] = time
        point[5] = mean
        tens = torch.tensor(point).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            AF = agent.actor(tens).cpu().numpy().squeeze()
            critic = agent.critic(tens[:, -2:]).cpu().numpy().squeeze()
        AFs[i, j] = AF
        critic_vals[i, j] = critic

fig, axs = plt.subplots(2, 1)

axs[0].set_title("AF")
img = axs[0].imshow(AFs)
plt.colorbar(img)
axs[0].invert_yaxis()
axs[0].set_ylabel("Time")
axs[0].set_xlabel("Mean")

"""axs[1].set_title("Critic")
img = axs[1].imshow(critic_vals)
plt.colorbar(img)
axs[1].invert_yaxis()
axs[1].set_ylabel("Time")
axs[1].set_xlabel("Mean")"""

plt.show()

plt.show()