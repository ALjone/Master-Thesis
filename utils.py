import torch

def rand(start, end, size) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, )
    return torch.distributions.uniform.Uniform(start, end).sample(size).to(torch.device("cuda")).squeeze()

def make_action(action, dims):
    """When doing batched stuff but only having an action for the first dim"""
    return torch.stack((torch.tensor(action), torch.tensor([0.12 for _ in range(dims)])), dim = 0).to(torch.device("cuda"))