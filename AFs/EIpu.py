import torch
import torch.nn as nn
from torch.distributions import Categorical
from scipy.stats import norm


class EIpu:

    def __init__(self, entropy = 0.001, dims = 2) -> None:
        self.entropy = torch.tensor([entropy])
        self.dims = dims

    def get_flattened_index(self, coords, resolution):
        """Convert coordinates to their corresponding flattened index."""
        dims = coords.size(1)

        # Calculate the stride for each dimension
        strides = torch.tensor([resolution ** (dims - i - 1) for i in range(dims)]).unsqueeze(0).to(coords.device)

        # Multiply the coordinates by the strides and sum along the dimensions to obtain the flattened index
        idx = torch.sum(coords * strides, dim=1, keepdim=True)

        return idx
    
    #Thanks to ChatGPT
    def get_coords(self, idx, dims, resolution):
        """Convert a flattened index to its corresponding coordinates."""
        coords = torch.zeros(idx.shape[0], dims, dtype=torch.int64, device=idx.device)

        if dims == 2:
            coords[:, 1] = idx % resolution
            coords[:, 0] = idx // resolution

        elif dims == 3:
            coords[:, 2] = idx % resolution
            idx = idx // resolution
            coords[:, 1] = idx % resolution
            coords[:, 0] = idx // resolution

        else:
            raise ValueError("dims should be 2 or 3")

        return coords

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.cpu()
        #Observation shape: [b, n, res * dim]
        #Mean, STD, timeportion in point, time spent (constant), best prediction globally (constant), max time
        assert observations.shape[1] == 6, "Needs to be ran in use_time format"
        max_val = observations[:, 4, 0, 0] if self.dims == 2 else observations[:, 4, 0, 0, 0]
        max_val = max_val.unsqueeze(1)
        mean = observations[:, 0]
        std = observations[:, 1]
        time = observations[:, 2]
        with torch.no_grad():
            biggest = torch.amax(max_val, dim=(1))
            for _ in range(self.dims):
                biggest = biggest.unsqueeze(1)
            e = self.entropy
            Z = (mean - biggest - e) / std

            EI = ((mean - biggest - e) * norm.cdf(Z) + std * norm.pdf(Z)).to(torch.float)

        return (EI/time).to(torch.device("cpu"))


    def get_value(self, x):
        _, value = self(x)
        return value

    def get_action_and_value(self, img, action=None, testing = False):
        logits = self(img)

        categorical = Categorical(logits = logits.flatten(1))
        action = categorical.mode
        indices_tensor = self.get_coords(action, self.dims, img.shape[-1])


        return indices_tensor, None, None, None
    