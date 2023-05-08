import torch

def rand(start, end, size) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, )
    return torch.distributions.uniform.Uniform(start, end).sample(size).to(torch.device("cpu")).squeeze()
