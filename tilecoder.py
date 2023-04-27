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

       

    def __getitem__(self, *args):
        x = args[0] #TODO: Why???
        output = []
        for elem, bin in zip(x, self.bins, strict=True):
            output.append(torch.bucketize(elem, bin))
        return output