import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand

class RandomGoldsteinPrice:
    def __init__(self, config) -> None:
        self.range_x1 = [-2, 2]  # range for x1 in the Goldstein-Price function
        self.range_x2 = [-2, 2]  # range for x2 in the Goldstein-Price function
        self.resolution = config.resolution    
        self.noise_scale = config.noise_scale
        self.batch_size = config.batch_size
        self.noise_correlation = config.noise_correlation
        self.dims = config.dims

        x1 = np.linspace(self.range_x1[0], self.range_x1[1], self.resolution)
        x2 = np.linspace(self.range_x2[0], self.range_x2[1], self.resolution)
        X1, X2 = np.meshgrid(x1, x2)
        X1 = np.repeat(X1[None, :, :], self.batch_size, axis=0)
        X2 = np.repeat(X2[None, :, :], self.batch_size, axis=0)
        self.x1 = torch.tensor(X1).to(torch.device("cpu"))
        self.x2 = torch.tensor(X2).to(torch.device("cpu"))

        # parameters for the Goldstein-Price function
        self.mean = 8.693
        self.std = 2.427

    def rotate(self, x, y, angles):
        x_rot = torch.cos(angles) * x - torch.sin(angles) * y
        y_rot = torch.sin(angles) * x + torch.cos(angles) * y
        return x_rot, y_rot

    def get_params(self, size):
        angles = (rand(0, 2*torch.pi, size = (size+1))[:-1, None, None])
        return {"angles": angles}
    
    def get_matrix(self, size):
        params = self.get_params(size)
        return self._make_matrix(size, params)
    
    def _get_noise(self, size):
        noise = np.random.normal(loc=1, scale=self.noise_scale, size=(tuple(self.resolution for _ in range(self.dims)) + (size, )))
        noise = gaussian_filter(noise, sigma=self.noise_correlation) # adjust sigma to control the amount of correlation

        return torch.tensor(noise).to(torch.device("cpu"))
    
    def _make_matrix(self, size, params):
        if self.dims == 3:
            return self._make_matrix_3d(size, params)
        if self.dims == 2:
            return self._make_matrix_2d(size, params)
        raise ValueError("Currently only supported for 2 and 3 dims")

    def _make_matrix_2d(self, size, params):
        X1_rot, X2_rot = self.rotate(self.x1[:size], self.x2[:size], params["angles"][:size])
        X1_rot, X2_rot = self.x1[:size], self.x2[:size]
        # Define the Goldstein-Price function
        gprice = (1 + (X1_rot + X2_rot + 1)**2 * (19 - 14*X1_rot + 3*X1_rot**2 - 14*X2_rot + 6*X1_rot*X2_rot + 3*X2_rot**2)) * \
                (30 + (2*X1_rot - 3*X2_rot)**2 * (18 - 32*X1_rot + 12*X1_rot**2 + 48*X2_rot - 36*X1_rot*X2_rot + 27*X2_rot**2))

        # Normalize the function
        gprice = 1 / self.std * (torch.log(gprice) - self.mean)
        # Maximize the function
        matrix = (-gprice).to(torch.float32)

        noise = self._get_noise(size).to(torch.float).squeeze()
        if size == 1:
            noise = noise.unsqueeze(2)
        noise = noise.permute(2, 0, 1)
        #matrix = matrix * noise
        return matrix


    def _make_matrix_3d(self, idx):
        return NotImplementedError("Rosenbrock functions are only implemented for 2d")

    def visualize_two_dims(self):
        matrix = self.get_matrix(10)
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].detach().cpu().numpy(), cmap='viridis')
            #ax.set_title('A = {:.2f}, B = {:.2f}, angle = {:.2f}'.format(self.params["a"][i].item(), self.params["b"][i].item(), self.params["angles"][i, 0, 0].item()))
        plt.tight_layout()
        plt.show() 