import torch        
import numpy as np
from env.batched_env import BlackBox
from agents.pix_2_pix_agent import Agent
from matplotlib import pyplot as plt

env = BlackBox(30, batch_size=2, dims = 2)
agent: Agent = Agent(env.observation_space, env.dims).to(torch.device("cuda"))
agent.load_state_dict(torch.load("model.t"))

s, t = env.reset()

prev_probs = np.zeros((env.resolution, env.resolution))
prev_logits = np.zeros((env.resolution, env.resolution))

for i in range(100):
    values = []
    done = False
    env.render()
    while not done:
        print("Time in env:", env.time[0].item(), "Max time:", env.T)
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(s, t)
        values.append(value.cpu().numpy()[0])
        _, _, dones, _ = env.step(action, True)
        print("Dones:", dones)
        done = dones[0]
    plt.close()
    plt.cla()
    plt.plot(values)
    plt.title("Critic guess as a function of t for the last seen state")
    plt.show()