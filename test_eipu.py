
from env.batched_env_pointwise import BlackBox
from baselines.baseline_agent import test_agent
#from AFs.EIpu import EIpu
from AFs.CArBO import CArBO as CArBO
import warnings
from utils import load_config, pretty_print_results, AttrDict
warnings.filterwarnings("ignore")

def dict_copy(d):
    return AttrDict({k:v for k,v in d.items()})

def baseline_EIpu(n, config, dims = 2, batch_size = None, entropy = 0.001):
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


n = 10000
batch_size = 1024
configs = configs = [
    "saved_configs\goldsteinprice_on_goldsteinprice_2d.yml",
    "saved_configs\goldsteinprice_on_multimodel_2d.yml",
    "saved_configs\goldsteinprice_training.yml",
    "saved_configs\multimodal_on_convex_2d.yml",
    "saved_configs\multimodal_on_goldsteinprice_2d.yml",
    "saved_configs\multimodal_on_multimodal_2d.yml",
    "saved_configs\multimodal_on_multimodal_3d_2_time_dims.yml",
    "saved_configs\multimodal_on_multimodal_3d_3_time_dims.yml",
    "saved_configs\multimodal_on_multimodal_different_time_1.yml",
    "saved_configs\multimodal_on_multimodal_different_time_2.yml",
    "saved_configs\multimodal_training.yml"
]
for conf in configs:
    config = load_config(conf)
    print("\n\n\nTesting on config:", conf[14:])
    print(f"\n\tEIpu agent with n = {n} sampling:")
    baseline_EIpu(n, config, config.dims, batch_size=batch_size)
