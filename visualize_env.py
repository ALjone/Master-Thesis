import torch        
from batched_env import BlackBox

env = BlackBox(40, (0, 10), 2, num_init_points=6, dims = 2, T = 100)

env.render()

for i in range(100):
    env.step(torch.tensor(env.action_space.sample()).to(torch.device("cuda")))
    env.reset()
    env.render()    