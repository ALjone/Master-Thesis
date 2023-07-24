#from env.batched_env import BlackBox
from env.batched_env_pointwise import BlackBox
from utils import load_config

config = load_config("configs\\training_config.yml", change_dict={"batch_size" : 2, "dims": 2, "num_init_points": 3})

env = BlackBox(config = config)

env.render()

for i in range(100):
    env.step(env.GP.get_next_point(return_idx = True))
    env.reset()
    env.render()