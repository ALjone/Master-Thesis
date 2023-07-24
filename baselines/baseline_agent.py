#Not technically baselining...
from agents.pointwise import Agent
from env.batched_env_pointwise import BlackBox
from tqdm import tqdm
import torch

def test_agent(env: BlackBox, agent: Agent, n: int):
    rewards = []
    lengths = []
    peaks = []
    actions = []

    s = env.reset()
    n_dones = 0
    with tqdm(total=n, desc = "Testing actor", leave = False) as pbar:
        while n_dones < n:
            act, _, _, _ = agent.get_action_and_value(s, testing = True)
            actions.append(torch.mean(act.to(torch.float32)).item())
            s, _, dones, info = env.step(act)
            if torch.sum(dones) > 0:
                rewards += info["episodic_returns"][dones].tolist()
                lengths += info["episodic_length"][dones].tolist()
                peaks += info["peak"][dones].tolist()
                pbar.update(torch.sum(dones).item())
                n_dones += torch.sum(dones).item()

    return rewards, lengths, peaks