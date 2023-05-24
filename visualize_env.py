import torch        
from batched_env import BlackBox

env = BlackBox(batch_size= 2, dims = 2, num_init_points=4)

env.render()

for i in range(100):
    env.step(env.GP.get_next_point(return_idx = False))
    env.reset()
    env.render()