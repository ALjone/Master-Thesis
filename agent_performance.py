from env.batched_env import BlackBox
from agents.pix_2_pix_agent import Agent
from tqdm import tqdm
from utils import load_config
import torch

def test_agent(env: BlackBox, agent: Agent, n: int):
    rewards = []
    lengths = []
    peaks = []
    actions = []

    s, t = env.reset()
    with tqdm(total=n, desc = "Baselining GPY", leave = False) as pbar:
        while len(peaks) < n:
            act, _ = agent.get_action_and_value(s, t)
            actions.append(torch.mean(act).item())
            (s, t), _, dones, info = env.step(act)
            if torch.sum(dones) > 0:
                rewards += info["episodic_returns"][dones].tolist()
                lengths += info["episodic_length"][dones].tolist()
                peaks += info["peak"][dones].tolist()
                pbar.update(torch.sum(dones).item())

    reward_avg = round(sum(rewards)/n, 4)
    length_avg = round(sum(lengths)/n, 4)
    peak_avg = round(sum(peaks)/n, 4)
    action_avg = round(sum(actions)/n, 4)

    reward_std = round(torch.std(rewards)/torch.sqrt(n), 4)
    length_std = round(torch.std(lengths)/torch.sqrt(n), 4)
    peak_std = round(torch.std(peaks)/torch.sqrt(n), 4)
    action_std = round(torch.std(actions)/torch.sqrt(len(actions)), 4)
    return reward_avg, length_avg, peak_avg, action_avg, reward_std, length_std, peak_std, action_std

    


training_config = load_config("C:\\Users\\Audun\\Thesis\\Reinforcement_Learning\\configs\\training_config.yml")
test_config = load_config("C:\\Users\\Audun\\Thesis\\Reinforcement_Learning\\configs\\testing_config.yml")

training_env = BlackBox(training_config)
test_env = BlackBox(test_config)

model = Agent(training_env.observation_space)
model.load_state_dict(torch.load(training_config.pre_trained_path))

n = 10

reward_avg, length_avg, peak_avg, action_avg, reward_std, length_std, peak_std, action_std = test_agent(training_config, model, n)
print("Agent's performance on training env:")
print(f"\tReward: {reward_avg} ± {reward_std}, Length: {length_avg} ± {length_std}, Peak: {peak_avg} ± {peak_std}, Action: {action_avg} ± {action_std}")

reward_avg, length_avg, peak_avg, action_avg, reward_std, length_std, peak_std, action_std = test_agent(test_config, model, n)

print("Agent's performance on testing env:")
print(f"\tReward: {reward_avg} ± {reward_std}, Length: {length_avg} ± {length_std}, Peak: {peak_avg} ± {peak_std}, Action: {action_avg} ± {action_std}")