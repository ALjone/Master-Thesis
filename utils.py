import yaml
import torch
from gpytorch.kernels import RBFKernel, MaternKernel
import numpy as np
#Thanks to GPT-4
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

def rand(start, end, size) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, )
    return torch.distributions.uniform.Uniform(start, end).sample(size).to(torch.device("cuda")).squeeze()

def make_action(action, dims):
    """When doing batched stuff but only having an action for the first dim"""
    return torch.stack((torch.tensor(action), torch.tensor([0.12 for _ in range(dims)])), dim = 0).to(torch.device("cuda"))

#Thanks to GPT-4
def metrics_per_class(rewards, peaks, lengths, class_idx, total_classes):
    mean_rewards = []
    mean_peaks = []
    mean_lengths = []
    for i in range(total_classes):
        if i in class_idx:
            mean_rewards.append(rewards[class_idx == i].mean().item())
            mean_peaks.append(peaks[class_idx == i].mean().item())
            mean_lengths.append(lengths[class_idx == i].mean().item())
        else:
            mean_rewards.append(None)
            mean_peaks.append(None)
            mean_lengths.append(None)

    return mean_rewards, mean_peaks, mean_lengths



def load_config(path: str, change_dict = {}):
    config = yaml.safe_load(open(path))

    for key, val in change_dict.items():
        config[key] = val

    for key, val in config.items():
        if isinstance(val, str) and val.lower() == "none":
            config[key] = None
    
    kernels = []
    for kernel in config["kernel_classes"]:
        assert kernel.lower() in ["rbf", "matern"], "Only RBF and Matern support atm"
        if kernel.lower() == "rbf":
            kernels.append(RBFKernel)
        elif kernel.lower() == "matern":
            kernels.append(MaternKernel)

    config["kernel_classes"] = kernels
    config = AttrDict(config)

    return config


def pretty_print_results(rewards, lengths, peaks, round_precision = 6):
    n = len(peaks)
    reward_avg = round(sum(rewards)/n, round_precision)
    length_avg = round(sum(lengths)/n, round_precision)
    peak_avg = round(sum(peaks)/n, round_precision)

    reward_error = round(np.std(rewards)/np.sqrt(n), round_precision)
    length_error = round(np.std(lengths)/np.sqrt(n), round_precision)
    peak_error = round(np.std(peaks)/np.sqrt(n), round_precision)
    
    print(f"\t\tReward: {reward_avg} ± {reward_error}, Length: {length_avg} ± {length_error}, Peak: {peak_avg} ± {peak_error}")
    print(f"\t\tLog-transformed simple regret: {round(-np.log10(1-peak_avg), round_precision)}, Simple regret: {round(1-peak_avg, round_precision)}")