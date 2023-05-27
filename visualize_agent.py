import torch        
import numpy as np
from batched_env import BlackBox
from agents.pix_2_pix_agent import Agent
from matplotlib import pyplot as plt

env = BlackBox(30, batch_size=2, dims = 2)
agent: Agent = Agent(env.observation_space, env.dims).to(torch.device("cuda"))
agent.load_state_dict(torch.load("model.t"))

s, t = env.reset()

prev_probs = np.zeros((env.resolution, env.resolution))
prev_logits = np.zeros((env.resolution, env.resolution))

for i in range(100):
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(s, t)
        logits, probs = agent.get_2d_logits_and_prob(s, t)
    normalize_probs = probs/probs[0].max()
    env.render(additional={"Agent probabilities": normalize_probs[0], "Agent logits": logits[0], "Previous probabilities": prev_probs, "Previous logits": prev_logits, "EI vs. probs" : env.grid[0, 2].cpu().numpy()-normalize_probs[0]})
    (s, t), _, _, _ = env.step(action, isindex = True)
    prev_probs = normalize_probs[0]
    prev_logits = logits[0]