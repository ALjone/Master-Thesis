from stable_baselines3 import PPO, A2C
from env import BlackBox
from callback import LoggerCallback
#from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from utils import benchmark, test_model, find_average_for_start_points

simulations = 10000

test_env = BlackBox()
average_start_points_string, start_point_average = find_average_for_start_points(test_env, simulations)
benchmark_string = benchmark(test_env, simulations, start_point_average)

# Parallel environments
env = make_vec_env(BlackBox, n_envs=24)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./PPO_SB3/", gamma = 1, batch_size = 1024)

#NOTE: Average starting point is 0.54187 (i.e 54% of max)

model.learn(total_timesteps=27648000, callback = LoggerCallback(), progress_bar = True)

print(average_start_points_string)
print(benchmark_string)
test_model(BlackBox(), sims = simulations, start_point_average = start_point_average)
model.save("Best")