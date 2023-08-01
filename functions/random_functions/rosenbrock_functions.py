import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand, load_config
#TODO:

#Troels talked about f(x, y, z) = a*f1(x)+b*f2(y)+c*f3(z)+d*f4(x, y, z) der f4(x, y, z) er en ikke linear kombinasjon der x y z er avhengig av hverandre, med en epsilon
#f1, f2, f3 etc kan være lavdimensionelle, med en NN som genererer litt woobliness 

class RandomRosenbrock:
    def __init__(self, config) -> None:
        self.range = [-2, 2]
        self.resolution = config.resolution    
        self.noise_scale = config.noise_scale
        self.batch_size = config.batch_size
        self.noise_correlation = config.noise_correlation
        self.a_vals = config.rosenbrock_a
        self.b_vals = config.rosenbrock_b
        self.dims = config.dims

        x = np.linspace(self.range[0], self.range[1], self.resolution)
        X, Y = np.meshgrid(x, x)
        X = np.repeat(X[None, :, :], self.batch_size, axis=0)
        Y = np.repeat(Y[None, :, :], self.batch_size, axis=0)
        self.x = torch.tensor(X).to(torch.device("cpu"))
        self.y = torch.tensor(Y).to(torch.device("cpu"))

    def get_params(self, size):
        #+1 to avoid the pesky size == 1
        a = -1*rand(self.a_vals[0], self.a_vals[1], size=(size+1))[:-1, None, None]
        b = rand(self.b_vals[0], self.b_vals[1], size=(size+1))[:-1, None, None]
        angles = (rand(0, 2*torch.pi, size = (size+1))[:-1, None, None])
        return {    "a": a,
                    "b": b,
                    "angles": angles}
    
    def get_matrix(self, size):
        params = self.get_params(size)
        return self._make_matrix(size, params)

    #Thanks to ChatGPT
    def rotate(self, x, y, angles):
        x_rot = torch.cos(angles) * x - torch.sin(angles) * y
        y_rot = torch.sin(angles) * x + torch.cos(angles) * y
        return x_rot, y_rot

    def _get_noise(self, size):
        noise = np.random.normal(loc=1, scale=self.noise_scale, size=(tuple(self.resolution for _ in range(self.dims)) + (size, )))
        noise = gaussian_filter(noise, sigma=self.noise_correlation) # adjust sigma to control the amount of correlation
        return torch.tensor(noise).to(torch.device("cpu"))
    
    def _make_matrix(self, size, params):
        if self.dims == 3:
            return self._make_matrix_3d(size, params)
        if self.dims == 2:
            return self._make_matrix_2d(size, params)
        raise ValueError("Currently only supported for 2 dims")
    
    def _make_matrix_2d(self, size, params):
        X_rot, Y_rot = self.rotate(self.x[:size], self.y[:size], params["angles"][:size])

        if size == 1:
            matrix = torch.log2(1 + 1 / ((params["a"] - X_rot) ** 2 + params["b"] * (Y_rot - X_rot**2) ** 2)).to(torch.float32)   
        else:
            matrix = torch.log2(1 + 1 / ((params["a"][:size] - X_rot) ** 2 + params["b"][:size] * (Y_rot - X_rot**2) ** 2)).to(torch.float32)
        noise = self._get_noise(size).to(torch.float).squeeze()
        if size == 1:
            noise = noise.unsqueeze(2)
        noise = noise.permute(2, 0, 1)
        matrix = matrix * noise
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

if __name__ == "__main__":
    #TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
    config = load_config("configs\\training_config.yml")
    config.domain = [-1, 1]
    f = RandomRosenbrock(config)
    for i in range(100):
        f.visualize_two_dims()
        f.reset()