from utils import pretty_print_results
from env.batched_env_pointwise import BlackBox
from agents.pointwise import Agent
from utils import load_config
import torch
from baselines.baseline_agent import test_agent

def baseline(training, n, model_path, use_time, dims = 2, batch_size = None):
    config = load_config("configs\\training_config.yml" if training else "configs\\testing_config.yml")
    if batch_size is not None:
        config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    config.use_time = use_time
    env = BlackBox(config)
    agent = Agent(env.observation_space, config.layer_size, dims = config.dims, verbose = False).to(torch.device("cuda"))
    agent.load_state_dict(torch.load(model_path))

    rewards, lengths, peaks = test_agent(env, agent, n)
    pretty_print_results(rewards, lengths, peaks)
    

if __name__ == "__main__":
    n = 20000
    model_path = "models\\test.t"

    print(f"Agent's performance on training env with dim = 2:")
    baseline(True, n, model_path, 2)

    print(f"Agent's performance on testing env with dim = 2:")
    baseline(False, n, model_path, 2)

    print(f"Agent's performance on training env with dim = 3:")
    baseline(True, n, model_path, 3, 64)