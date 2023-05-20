import torch        
from batched_env import BlackBox
from tanh_agent import Agent as tanh_agent

env = BlackBox(30, (0, 10), 2, num_init_points=2, dims = 2, T = 60)
agent: tanh_agent = torch.load("model.t")

s, t = env.reset()

for i in range(100):
    with torch.no_grad():
        action, lp, _, _, _ = agent.get_action_and_value(s, t)
        print("Prob:", round(torch.exp(lp[0]).item(), 3), "Log prob:", round(lp[0].item(), 3))
        agent.visualize_dist(s, t)
    (s, t), _, _, _ = env.step(action, True)
    env.render()    