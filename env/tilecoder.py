import numpy as np
import torch

class TileCoder:
    def __init__(self, resolution, domain, dims = 3) -> None:
        assert len(domain) == 2
        x_min = domain[0]
        x_max = domain[1]
        #NO TILE CODING WITH THIS:
        self.bins = []
        for _ in range(dims):
            self.bins.append(torch.linspace(x_min, x_max, resolution).to(torch.device("cuda")))

       

    def __getitem__(self, x):
        output = []
        for i, bin in enumerate(self.bins):
            output.append(torch.bucketize(x[:, i], bin))
        return torch.stack(output, dim = 1).to(torch.device("cuda"))
    

    if __name__ == "__main__":
        pass
        #TODO: Test!