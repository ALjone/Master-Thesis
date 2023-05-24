import torch        
from batched_env import BlackBox

env = BlackBox(batch_size= 2, dims = 2, num_init_points=3)

env.render()

for i in range(100):
    env.step(torch.tensor(env.action_space.sample()).to(torch.device("cuda")))
    #env.reset()
    env.render()