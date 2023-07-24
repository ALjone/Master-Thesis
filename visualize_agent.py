import torch        
import numpy as np
#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
#from agents.pix_2_pix_agent import Agent
#from agents.ei_argmax_weighting import Agent
from agents.pointwise import Agent
from utils import load_config

config = load_config("configs\\training_config.yml")
config.dims = 2
config.batch_size = 2
#config.resolution = 100

env = BlackBox(config)
agent: Agent = Agent(env.observation_space, env.dims).to(torch.device("cuda"))
#agent.load_state_dict(torch.load("model_with_positional_encoding.t"))
#agent.load_state_dict(torch.load("models/With positional encoding, temperature, convex, relu, high res.t"))
agent.load_state_dict(torch.load("models/convex only new eval cost.t"))

s = env.reset()

prev_probs = np.zeros((env.resolution, env.resolution))
prev_logits = np.zeros((env.resolution, env.resolution))    

for i in range(100):
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(s, testing=True)
        logits, probs = agent.get_2d_logits_and_prob(s)
    print(logits[0].max())
    print(probs[0].max())
    env.render(additional={"Agent probabilities": probs[0], "Agent logits": logits[0], "Previous probabilities": prev_probs, "Previous logits": prev_logits, "EI vs. probs" : env.grid[0, 2].cpu().numpy()-probs[0]})
    agent.visualize_positional_weighting()
    prev_probs = probs[0]
    prev_logits = logits[0]
    s, _, _, _ = env.step(action, isindex = True)