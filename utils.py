from env import BlackBox
import numpy as np
from tqdm import tqdm
import gym
from stable_baselines3 import PPO

def find_average_for_start_points(env: gym.Env, sims = 10000):

    R = 0
    for _ in tqdm(range(sims), leave=False, desc="Average value"):
        env.reset()
        R += env.previous_closeness_to_max

    string = f"Initial points:\n\tAverage value from 2 initial points: {round(R/sims, 3)}" + "\n"
    print(string)
    print()
    return string, R/sims

def benchmark(env: gym.Env, sims = 10000, start_point_average: float = 0):
    R = 0
    t = 0

    for _ in tqdm(range(sims), leave=False, desc = "Benchmark"):
        env.reset()

        done = False
        while not done:
            _, r, done, _ = env.step(env.action_space.sample())
            R += r
            t += 1

    bench_string = f"Benchmark with random agent:\n\tAverage reward: {round(R/sims, 3)}\n\tAverage steps: {round(t/sims, 3)}" + f"\n\tAverage closeness to max with init point average: {round((R/sims) + start_point_average, 3)}" if start_point_average > 0 else " " + "\n"
    print(bench_string)
    return bench_string

def test_model(env: gym.Env, model: PPO = None, sims = 1000, start_point_average: float = 0):
    if model is None:
        model = PPO.load("Best")

    R = 0
    steps = 0
    for _ in tqdm(range(sims), leave=False, desc = "Testing model"):
        s = env.reset()

        done = False
        while not done:
            action, _ = model.predict(s, deterministic = True)
            s, r, done, _ = env.step(action)
            R += r
            steps += 1

    eval_string = f"Evaluation:\n\tAverage reward: {round(R/sims, 3)}\n\tAverage steps: {round(steps/sims, 3)}"  + f"\n\tAverage closeness to max with init point average: {round((R/sims) + start_point_average, 3)}" if start_point_average > 0 else " " + "\n"
    print(eval_string)
    return eval_string

if __name__ == "__main__":
    test_model(BlackBox(), sims = 10000)