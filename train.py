from stable_baselines3 import PPO
from game import BlackBox

from stable_baselines3.common.env_util import make_vec_env
from benchmark import benchmark

benchmark(1000)

# Parallel environments
env = make_vec_env(BlackBox, n_envs=4)
    
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./PPO_SB3/", gamma = 1)
#model.learn(total_timesteps=500000, progress_bar=True)
#model = PPO.load("Best", env)

#model.learn(total_timesteps=15000000, progress_bar=True)
model.learn(total_timesteps=1500000, progress_bar=True)
model.save("Best")