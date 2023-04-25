import numpy as np
import torch

class TileCoder:
    def __init__(self, resolution, domain) -> None:
        x_min = domain[0]
        x_max = domain[1]
        y_min = domain[0]
        y_max = domain[1]
        z_min = domain[0]
        z_max = domain[1]
        #NO TILE CODING WITH THIS:

        self.x_bins = torch.linspace(x_min, x_max, resolution).to(torch.device("cuda"))
        self.y_bins = torch.linspace(y_min, y_max, resolution).to(torch.device("cuda"))
        self.z_bins = torch.linspace(z_min, z_max, resolution).to(torch.device("cuda"))

       

    def __getitem__(self, x):
        #TODO: Needs to work with batches?
        x, y, z = x
        x = torch.bucketize(x, self.x_bins)
        y = torch.bucketize(y, self.y_bins)
        z = torch.bucketize(z, self.z_bins)
        return (x, y, z)