import yaml
import torch
from gpytorch.kernels import RBFKernel, MaternKernel

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


def load_config(path: str, change_dict = {}):
    config = yaml.safe_load(open(path))

    for key, val in change_dict.items():
        config[key] = val

    for key, val in config.items():
        if isinstance(val, str) and val.lower() == "none":
            config[key] = None
    
    kernels = []
    for kernel in config["kernels"]:
        assert kernel.lower() in ["rbf", "matern"], "Only RBF and Matern support atm"
        if kernel.lower() == "rbf":
            kernels.append(RBFKernel)
        elif kernel.lower() == "matern":
            kernels.append(MaternKernel)

    config["kernels"] = kernels
    config = AttrDict(config)

    return config