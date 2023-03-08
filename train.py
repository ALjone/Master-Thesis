from stable_baselines3 import PPO
from env import BlackBox
#from stable_baselines3.common.callbacks import EvalCallback, CallbackList
#from stable_baselines.common.callbacks import CheckpointCallback
from callback import LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from utils import benchmark, test_model, find_average_for_start_points
from network import CustomCNN

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

simulations = 10

test_env = BlackBox()
average_start_points_string, start_point_average = find_average_for_start_points(test_env, simulations)
benchmark_string = benchmark(test_env, simulations, start_point_average)

# Parallel environments
env = make_vec_env(BlackBox, n_envs=3)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./PPO_SB3/", gamma = 1, batch_size = 128, policy_kwargs=policy_kwargs)

#NOTE: Average starting point is 0.54187 (i.e 54% of max)
model.learn(total_timesteps=276480, callback = [LoggerCallback()], progress_bar = True, log_interval=1)
model.save("Best")
for i in range(100000):
    model.load("Best")
    model.learn(total_timesteps=276480, callback = [LoggerCallback()], progress_bar = True, log_interval=1   )
    model.save("Best")

print(average_start_points_string)
print(benchmark_string)
test_model(BlackBox(), sims = simulations, start_point_average = start_point_average)