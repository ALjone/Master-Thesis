import torch        
from batched_env import BlackBox
from agents.tanh_agent import Agent as tanh_agent
from matplotlib import pyplot as plt

env = BlackBox(30, batch_size=2, dims = 2)
agent: tanh_agent = torch.load("Pretrained_tanh_agent_2.t")

s, t = env.reset()

for i in range(100):
    with torch.no_grad():
        action, lp, _, _, _ = agent.get_action_and_value(s, t)
        #print("Prob:", round(torch.exp(lp[0]).item(), 3), "Log prob:", round(lp[0].item(), 3))
        action_mean, action_std, _ = agent(s, t)
        #action_mean = torch.tensor([[-1, -1], [-1, -1]]).to(torch.device("cuda"))
    fig, axs = env.render(tanh_mean = torch.tanh(action_mean[0]).unsqueeze(0))    
    (s, t), _, _, _ = env.step(action, True)
    #s, t = env.reset()