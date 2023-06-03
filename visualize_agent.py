import torch        
import numpy as np
from env.batched_env import BlackBox
from agents.pix_2_pix_agent import Agent
from matplotlib import pyplot as plt
from utils import load_config

config = load_config("configs\\training_config.yml")

env = BlackBox(config)
agent: Agent = Agent(env.observation_space, env.dims).to(torch.device("cuda"))
#agent.load_state_dict(torch.load("model_with_positional_encoding.t"))
agent.load_state_dict(torch.load("models/model.t"))

s, t = env.reset()

prev_probs = np.zeros((env.resolution, env.resolution))
prev_logits = np.zeros((env.resolution, env.resolution))

for i in range(100):
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(s, t)
        logits, probs = agent.get_2d_logits_and_prob(s, t)
    print(logits[0].max())
    print(probs[0].max())
    print("Temperature:", agent.temperature.item())
    normalize_probs = probs/probs[0].max()
    env.render(additional={"Agent probabilities": normalize_probs[0], "Agent logits": logits[0], "Previous probabilities": prev_probs, "Previous logits": prev_logits, "EI vs. probs" : env.grid[0, 2].cpu().numpy()-normalize_probs[0]})
    agent.visualize_positional_weighting()
    (s, t), _, _, _ = env.step(action, isindex = True)
    prev_probs = normalize_probs[0]
    prev_logits = logits[0]