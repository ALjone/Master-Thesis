import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand

class RandomMultimodal:
    def __init__(self, config) -> None:
        self.range = config.domain
        self.resolution = config.resolution    
        self.noise_scale = config.noise_scale
        self.batch_size = config.batch_size
        self.noise_correlation = config.noise_correlation
        self.params = {}
        self.a_vals = config.convex_a
        self.b_vals = config.convex_b
        self.dims = config.dims
            
        x = torch.linspace(self.range[0], self.range[1], self.resolution)
        self.x = torch.repeat_interleave(x, self.batch_size).reshape(-1, self.batch_size).to(torch.device("cpu"))

    def get_params(self, size):
        params = {}
        amplitude_range = [0.8, 1.2]
        decay_range = [0.15, 0.5]
        mean_range = [[-0.85, -0.15], [0.15, 0.85]]  # Ensure the means lie within each quadrant

        # For each dimension, generate parameters for two Gaussians
        for dim in range(self.dims):
            amplitude1 = rand(amplitude_range[0], amplitude_range[1], size=(size))
            decay1 = rand(decay_range[0], decay_range[1], size=(size))
            mean1 = rand(mean_range[0][0], mean_range[0][1], size=(size))

            amplitude2 = rand(amplitude_range[0], amplitude_range[1], size=(size))
            decay2 = rand(decay_range[0], decay_range[1], size=(size))
            mean2 = rand(mean_range[1][0], mean_range[1][1], size=(size))

            params[dim] = { "amplitude1": amplitude1, "decay1": decay1, "mean1": mean1,
                            "amplitude2": amplitude2, "decay2": decay2, "mean2": mean2}

        return params
    
    def get_matrix(self, size):
        params = self.get_params(size)
        return self._make_matrix(size, params)

    def f(self, params, x, size):
        if size == 1:
            gaussian1 = params["amplitude1"] * torch.exp(-((x[:, :size] - params["mean1"]) ** 2) / (2 * params["decay1"] ** 2))
            gaussian2 = params["amplitude2"] * torch.exp(-((x[:, :size] - params["mean2"]) ** 2) / (2 * params["decay2"] ** 2))
        else:
            gaussian1 = params["amplitude1"][:size] * torch.exp(-((x[:, :size] - params["mean1"][:size]) ** 2) / (2 * params["decay1"][:size] ** 2))
            gaussian2 = params["amplitude2"][:size] * torch.exp(-((x[:, :size] - params["mean2"][:size]) ** 2) / (2 * params["decay2"][:size] ** 2))
        
        # Return sum of the two Gaussian functions
        return (gaussian1 + gaussian2 + -1 * torch.amin(gaussian1+gaussian2, dim=0))


    
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
        if size == 1:
            matrix = matrix.unsqueeze(-1)

        return matrix.permute(2, 0, 1).to(torch.float)

    def _make_matrix_3d(self, size, params):
        f_1 = self.f(params[0], self.x, size)
        f_2 = self.f(params[1], self.x, size)
        f_3 = self.f(params[2], self.x, size)

        if size > 1:
            matrix = torch.einsum('ib,jb,kb->ijkb', f_1, f_2, f_3)
        else:
            matrix = torch.einsum('i,j,k->ijk', f_1.squeeze(), f_2.squeeze(), f_3.squeeze())
        if size == 1:
            matrix = matrix.unsqueeze(-1)


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
