import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand, load_config

class LinearTime:
    def __init__(self, config) -> None:
        self.range = config.domain
        self.resolution = config.resolution    
        self.batch_size = config.batch_size
        self.params = {}
        self.coeffs = config.linear_range
        self.constant = config.constant
        self.dims = config.dims
        self.max_time = config.max_range
            
        x = torch.linspace(self.range[0], self.range[1], self.resolution)
        self.x = torch.repeat_interleave(x, self.batch_size).reshape(-1, self.batch_size).to(torch.device("cuda"))

    def get_params(self, size):
        params = {}
        for dim in range(self.dims):
            coeff = -1*rand(self.coeffs[0], self.coeffs[1], size=(size))
            c = rand(self.constant[0], self.constant[1], size=(size))
            max_time = rand(self.max_range[0], self.max_range[1], size=(size))
            params[dim] = { "coeff": coeff,
                            "c": c,
                            "max_time": max_time}
        return params
    
    def get_matrix(self, size):
        params = self.get_params(size)
        return self._make_matrix(size, params)

    def f(self, params, x, size):
        if size == 1:
            f = params["coeff"]*x[:, :size] #(params["a"] * ((x[:, :size] - params["p"])**2) * torch.exp(-params["b"] * (x[:, :size] - params["p"])))
        else:
            f = params["coeff"][:size]*x[:, :size] #(params["a"][:size] * ((x[:, :size] - params["p"][:size])**2) * torch.exp(-params["b"][:size] * (x[:, :size] - params["p"][:size])))
        return (f + -1*torch.amin(f, dim  = 0))

    def _make_matrix(self, size, params):
        if self.dims == 3:
            return self._make_matrix_3d(size, params)
        if self.dims == 2:
            return self._make_matrix_2d(size, params)
        raise ValueError("Currently only supported for 2 and 3 dims")
    
    def _make_matrix_2d(self, size, params):
        f_1 = self.f(params[0], self.x, size)
        f_2 = self.f(params[1], self.x, size)

        if size > 1:
            matrix = torch.einsum('ib,jb->ijb', f_1, f_2)
        else:
            matrix = torch.einsum('i,j->ij', f_1.squeeze(), f_2.squeeze())

        matrix += self.constant
        
        matrix += -1*(torch.amin(matrix, dim = (0, 1)))
        matrix = matrix/torch.amax(matrix, dim = (0, 1))
        if size == 1:
            matrix = matrix.unsqueeze(2)
        return matrix.permute(2, 0, 1)

    def _make_matrix_3d(self, size, params):
        f_1 = self.f(params[0], self.x, size)
        f_2 = self.f(params[1], self.x, size)
        f_3 = self.f(params[2], self.x, size)

        if size > 1:
            matrix = torch.einsum('ib,jb,kb->ijkb', f_1, f_2, f_3)
        else:
            matrix = torch.einsum('i,j,k->ijk', f_1.squeeze(), f_2.squeeze(), f_3.squeeze())
        noise = self._get_noise(size).to(torch.float).squeeze()

        matrix = matrix * noise
        matrix = matrix/torch.amax(matrix, dim = (0, 1, 2))
        return matrix.permute(3, 0, 1, 2)
    

    
    def visualize_two_dims(self):
        matrix = self.get_matrix(self.batch_size)
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"
        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].cpu().numpy(), cmap='viridis')
        plt.tight_layout()
        plt.show() 


if __name__ == "__main__":
    config = load_config("configs\\training_config.yml")
    f = LinearTime(config)
    for i in range(100):
        f.visualize_two_dims()
        f.reset()