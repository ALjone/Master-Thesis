from copy import deepcopy
from baselines.baseline_random import run as run_random
from baselines.baseline_gpy import run as run_gpy
from env.batched_env_pointwise import BlackBox
from agents.pointwise import Agent
from baselines.baseline_agent import test_agent
from AFs.EIpu import EIpu
from AFs.CArBO import CArBO
import torch
import warnings
from utils import load_config, pretty_print_results, AttrDict
warnings.filterwarnings("ignore")

def dict_copy(d):
    return AttrDict({k:v for k,v in d.items()})

def baseline_agent(n, model_path, use_time, config, dims = 2, batch_size = None, testing = True):
    config = dict_copy(config)
    if batch_size is not None:
        config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    config.use_time = use_time
    env = BlackBox(config)
    agent = Agent(env.observation_space, config.layer_size, dims = config.dims, verbose = False).to(torch.device("cpu"))
    agent.load_state_dict(torch.load(model_path))

    rewards, lengths, peaks = test_agent(env, agent, n, testing = testing)
    pretty_print_results(rewards, lengths, peaks)

def baseline_EIpu(n, config, dims = 2, batch_size = None, entropy = 0.001):
    config = dict_copy(config)
    if batch_size is not None:
        config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    config.use_time = True
    env = BlackBox(config)
    agent = EIpu(entropy = entropy, dims = config.dims)

    rewards, lengths, peaks = test_agent(env, agent, n, testing = True)
    pretty_print_results(rewards, lengths, peaks)

def baseline_CArBO(n, config, dims = 2, batch_size = None, entropy = 0.001):
    config = dict_copy(config)
    if batch_size is not None:
        config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    config.use_time = True
    env = BlackBox(config)
    agent = CArBO(entropy = entropy, dims = config.dims)

    rewards, lengths, peaks = test_agent(env, agent, n, testing = True)
    pretty_print_results(rewards, lengths, peaks)


def baseline(run, n, config, dims = 2, batch_size = 512):
    config = dict_copy(config)
    config.batch_size = batch_size
    config.dims = dims
    config.verbose = 0
    rewards, lengths, peaks = run(n, config)
    pretty_print_results(rewards, lengths, peaks)


n = 5000

time_models = ["models\\goldsteinprice temperature.t",
               #"models\\goldsteinprice temperature.t"
               "models\\multimodal temperature.t",
               #"models\\multimodal temperature.t",
               #"models\\multimodal temperature.t",
               #"models\\multimodal temperature.t",
               #"models\\multimodal temperature.t",
               #"models\\multimodal temperature.t",
               #"models\\multimodal temperature.t"
               ]

configs = [
    #"saved_configs\goldsteinprice_on_goldsteinprice_2d.yml",
    "saved_configs\goldsteinprice_on_multimodel_2d.yml",
    #"saved_configs\multimodal_on_convex_2d.yml",
    #"saved_configs\multimodal_on_goldsteinprice_2d.yml",
    #"saved_configs\multimodal_on_multimodal_2d.yml",
    #"saved_configs\multimodal_on_multimodal_3d_2_time_dims.yml",
    #"saved_configs\multimodal_on_multimodal_3d_3_time_dims.yml",
    #"saved_configs\multimodal_on_multimodal_different_time_1.yml",
    "saved_configs\multimodal_on_multimodal_different_time_2.yml"
]
for conf, time_model in zip(configs, time_models, strict=True):
    config = load_config(conf)
    no_time_model = time_model[:-2] + " no time.t"

    print(f"\n\nBaselining training env on {conf}\n")


    print(f"\tBenchmark random with n = {n}:")
    baseline(run_random, n, config, config.dims, batch_size=config.batch_size)
    print(f"\n\tBenchmark gpy with n = {n}:")
    baseline(run_gpy, n, config, config.dims, batch_size=config.batch_size)
    print(f"\n\tBenchmark EIpu with n = {n} sampling:")
    baseline_EIpu(n, config, config.dims, batch_size=config.batch_size)
    print(f"\n\tBenchmark CArBO with n = {n} sampling:")
    baseline_CArBO(n, config, config.dims, batch_size=config.batch_size)


    #print(f"\n\tTime agent with n = {n} using argmax:")
    #baseline_agent(n, time_model, True, config, config.dims, batch_size=config.batch_size, testing=True)
    #print(f"\n\tNo time agent with n = {n} using argmax:")
    #baseline_agent(n, no_time_model, False, config, config.dims, batch_size=config.batch_size, testing=True)

    #print(f"\n\tTime agent with n = {n} sampling:")
    #baseline_agent(n, time_model, True, config, config.dims, batch_size=config.batch_size, testing=False)
    #print(f"\n\tNo time agent with n = {n} sampling:")
    #baseline_agent(n, no_time_model, False, config, config.dims, batch_size=config.batch_size, testing=False)
