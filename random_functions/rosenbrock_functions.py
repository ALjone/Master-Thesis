import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand, load_config
#TODO:

#Troels talked about f(x, y, z) = a*f1(x)+b*f2(y)+c*f3(z)+d*f4(x, y, z) der f4(x, y, z) er en ikke linear kombinasjon der x y z er avhengig av hverandre, med en epsilon
#f1, f2, f3 etc kan være lavdimensionelle, med en NN som genererer litt woobliness 

class RandomFunction:
    def __init__(self, config) -> None:
        self.range = config.domain
        self.resolution = config.resolution    
        self.noise_scale = config.noise_scale
        self.batch_size = config.batch_size
        self.noise_correlation = config.noise_correlation
        self.params = {}
        self.a_vals = config.rosenbrock_a
        self.b_vals = config.rosenbrock_b
        self.dims = config.dims
        self.params= {  "a": -1*rand(self.a_vals[0], self.a_vals[1], size=(self.batch_size))[:, None, None],
                        "b": rand(self.b_vals[0], self.b_vals[1], size=(self.batch_size))[:, None, None],
                        "angles":  (torch.pi*torch.rand(self.batch_size))[:, None, None].to(torch.device("cuda"))}
            
        x = np.linspace(self.range[0], self.range[1], self.resolution)
        X, Y = np.meshgrid(x, x)
        X = np.repeat(X[None, :, :], self.batch_size, axis=0)
        Y = np.repeat(Y[None, :, :], self.batch_size, axis=0)
        self.x = torch.tensor(X).to(torch.device("cuda"))
        self.y = torch.tensor(Y).to(torch.device("cuda"))

        self.zeros = torch.zeros((self.batch_size)).to(torch.device("cuda"))

        self.matrix = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cuda"))
        self.max = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        self.min = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        
        self.reset()


    def reset(self, idx = None):
        if idx is None: idx = torch.arange(start = 0, end = self.batch_size)
        a = self.params["a"]
        b = self.params["b"]
        angles = self.params["angles"]

        a[idx] = -1*rand(self.a_vals[0], self.a_vals[1], size=(idx.shape[0]))[:, None, None]
        b[idx] = rand(self.b_vals[0], self.b_vals[1], size=(idx.shape[0]))[:, None, None]
        angles[idx] = (torch.pi*torch.rand(idx.shape[0]))[:, None, None].to(torch.device("cuda"))
        self.params = {     "a": a,
                            "b": b,
                            "angles": angles}
        self._make_matrix(idx)
            
    def f(self, params, x, idx):
        f = (params["a"][idx] * ((x[:, idx] - params["p"][idx])**2) * torch.exp(-params["b"][idx] * (x[:, idx] - params["p"][idx])))
        return (f + -1*torch.amin(f, dim  = 0))

    #Thanks to ChatGPT
    def rotate(self, x, y, angles):
        x_rot = torch.cos(angles) * x - torch.sin(angles) * y
        y_rot = torch.sin(angles) * x + torch.cos(angles) * y
        return x_rot, y_rot

    def _get_noise(self, idx):
        noise = np.random.normal(loc=1, scale=self.noise_scale, size=(tuple(self.resolution for _ in range(self.dims)) + (idx.shape[0], )))
        noise = gaussian_filter(noise, sigma=self.noise_correlation) # adjust sigma to control the amount of correlation
        return torch.tensor(noise).to(torch.device("cuda"))
    
    def _make_matrix(self, idx):
        if self.dims == 3:
            return self._make_matrix_3d(idx)
        if self.dims == 2:
            return self._make_matrix_2d(idx)
        raise ValueError("Currently only supported for 2 dims")
    
    def _make_matrix_2d(self, idx):

        X_rot, Y_rot = self.rotate(self.x[idx], self.y[idx], self.params["angles"][idx])
        matrix = torch.log10(1 + 1 / ((self.params["a"][idx] - X_rot) ** 2 + self.params["b"][idx] * (Y_rot - X_rot**2) ** 2)).to(torch.float32)        
        noise = self._get_noise(idx).to(torch.float).squeeze().permute(2, 0, 1)
        matrix = matrix * noise
        matrix += -1*(torch.amin(matrix, dim = (1, 2))[:, None, None])
        matrix = matrix/torch.amax(matrix, dim = (1, 2))[:, None, None]
        self.max[idx] = torch.ones((len(idx), )).to(torch.device("cuda"))
        assert len(self.max.shape) == 1, f"Expected max to have one dimension, found: {self.max.shape}"
        self.min[idx] = torch.amin(matrix, dim = (1, 2)).to(torch.float32)


        self.matrix[idx] = matrix.squeeze()

        assert torch.min(self.min) >= 0, f"Min shouldn't be less than 0, found: {torch.min(self.min)}"
        assert torch.max(self.max) <= 1, f"Max shouldn't be bigger than 1, found: {torch.max(self.max)}"
        assert self.matrix.shape[0] == self.batch_size, f"Expected matrix to have leading dimension of size {self.batch_size}, but found: {self.matrix.shape[0]}. Whole shape:{self.matrix.shape} "
    
    def _make_matrix_3d(self, idx):
        return NotImplementedError("Rosenbrock functions are only implemented for 2d")

    def visualize_two_dims(self):
        matrix = self.matrix
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(self.matrix[i].detach().cpu().numpy(), cmap='viridis')
            ax.set_title('A = {:.2f}, B = {:.2f}, angle = {:.2f}'.format(self.params["a"][i].item(), self.params["b"][i].item(), self.params["angles"][i, 0, 0].item()))
        plt.tight_layout()
        plt.show() 

if __name__ == "__main__":
    #TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
    config = load_config("configs\\training_config.yml")
    f = RandomFunction(config)
    for i in range(100):
        f.visualize_two_dims()
        f.reset()