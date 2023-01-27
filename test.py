from stable_baselines3.common.env_util import make_vec_env
from env import BlackBox
env = make_vec_env(BlackBox, n_envs=4)

