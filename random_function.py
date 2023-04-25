import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import rand
#TODO:

#Troels talked about f(x, y, z) = a*f1(x)+b*f2(y)+c*f3(z)+d*f4(x, y, z) der f4(x, y, z) er en ikke linear kombinasjon der x y z er avhengig av hverandre, med en epsilon
#f1, f2, f3 etc kan være lavdimensionelle, med en NN som genererer litt woobliness 

class RandomFunction:
    def __init__(self, domain, resolution, batch_size, noise_scale = 2, noise_correlation = 1) -> None:
        self.range = domain
        self.resolution = resolution    
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.noise_correlation = noise_correlation
        self.params = {}
        self.a_vals = (0.01, 0.5)
        self.b_vals = (0.01, 0.5)
        for dim in range(3):
            self.params[dim] = {"a": -1*rand(self.a_vals[0], self.a_vals[1], size=(self.batch_size)),
                                "b": rand(self.b_vals[0], self.b_vals[1], size=(self.batch_size)),
                                "p": rand(self.range[0], self.range[1], size=(self.batch_size))}
            
        x = torch.linspace(self.range[0], self.range[1], self.resolution)
        self.x = torch.repeat_interleave(x, self.batch_size).reshape(-1, self.batch_size).to(torch.device("cuda"))
        self.zeros = torch.zeros((self.batch_size)).to(torch.device("cuda"))

        self.matrix = torch.zeros((self.batch_size, self.resolution, self.resolution, self.resolution)).to(torch.device("cuda"))
        self.max = torch.zeros((batch_size)).to(torch.device("cuda"))
        self.min = torch.zeros((batch_size)).to(torch.device("cuda"))
        
        self.reset()


    def reset(self, idx = None):
        if idx is None: idx = torch.arange(start = 0, end = self.batch_size)
        for dim in range(3):
            a = self.params[dim]["a"]
            b = self.params[dim]["b"]
            p = self.params[dim]["p"]

            a[idx] = -1*rand(self.a_vals[0], self.a_vals[1], size=(idx.shape[0]))
            b[idx] = rand(self.b_vals[0], self.b_vals[1], size=(idx.shape[0]))
            p[idx] = rand(self.range[0], self.range[1], size=(idx.shape[0]))
            self.params[dim] = {"a": a,
                                "b": b,
                                "p": p}
        self._make_matrix(idx)
            
    def f(self, params, x, idx):
        f = (params["a"][idx] * ((x[:, idx] - params["p"][idx])**2) * torch.exp(-params["b"][idx] * (x[:, idx] - params["p"][idx])))
        return (f + -1*torch.amin(f, dim  = 0))

    def get_noise(self, idx):
        noise = np.random.normal(loc=1, scale=self.noise_scale, size=(self.resolution, self.resolution, self.resolution, idx.shape[0]))
        noise = gaussian_filter(noise, sigma=self.noise_correlation) # adjust sigma to control the amount of correlation

        return torch.tensor(noise).to(torch.device("cuda"))
    
    def _make_matrix(self, idx):
        f_1 = self.f(self.params[0], self.x, idx)
        f_2 = self.f(self.params[1], self.x, idx)
        f_3 = self.f(self.params[2], self.x, idx)

        if idx.shape[0] > 1:
            matrix = torch.einsum('ib,jb,kb->ijkb', f_1, f_2, f_3) #np.outer(f_1, f_2) 
        else:
            matrix = torch.einsum('i,j,k->ijk', f_1.squeeze(), f_2.squeeze(), f_3.squeeze()) #np.outer(f_1, f_2)     
        noise = self.get_noise(idx).to(torch.float).squeeze()

        matrix = matrix * noise
        matrix = matrix/torch.amax(matrix, dim = (0, 1, 2))
        self.max[idx] = torch.ones((len(idx), )).to(torch.device("cuda"))#torch.amax(matrix, dim = (0, 1, 2))
        assert len(self.max.shape) == 1, f"Expected max to have one dimension, found: {self.max.shape}"
        self.min[idx] = torch.amin(matrix, dim = (0, 1, 2))


        if idx.shape[0] == 1:
            self.matrix[idx] = matrix.squeeze()
        else:
            self.matrix[idx] = matrix.permute(3, 0, 1, 2)

        assert self.matrix.shape[0] == self.batch_size, f"Expected matrix to have leading dimension of size {self.batch_size}, but found: {self.matrix.shape[0]}. Whole shape:{self.matrix.shape} "

    def visualize_two_dims(self):
        matrix = self.matrix#.squeeze()[:, :, 0]

        for i in range(self.batch_size):
            dim_0 = ", ".join([f"{key}: {round(item[i].item(), 4)}" for key, item in self.params[0].items()])
            dim_1 = ", ".join([f"{key}: {round(item[i].item(), 4)}" for key, item in self.params[1].items()])
            dim_2 = ", ".join([f"{key}: {round(item[i].item(), 4)}" for key, item in self.params[2].items()])
            plt.title(f"Dim 0: {dim_0} \nDim 1: {dim_1}, \nDim 2: {dim_2}\nDim 0")
            plt.imshow(torch.mean(matrix[i], dim = 0).squeeze().cpu().numpy())
            plt.colorbar()
            plt.show()
            plt.cla()
            plt.title(f"Dim 0: {dim_0} \nDim 1: {dim_1}, \nDim 2: {dim_2}\nDim 1")
            plt.imshow(torch.mean(matrix[i], dim = 1).squeeze().cpu().numpy())
            plt.colorbar()
            plt.show()
            plt.cla()
            plt.title(f"Dim 0: {dim_0} \nDim 1: {dim_1}, \nDim 2: {dim_2}\nDim 2") 
            plt.imshow(torch.mean(matrix[i], dim = 2).squeeze().cpu().numpy())
            plt.colorbar()
            plt.show()
            plt.cla()
            print(i)


if __name__ == "__main__":
    #TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
    f = RandomFunction((0, 10), 40, 2)
    for i in range(100):
        f.visualize_two_dims()
        f.reset()