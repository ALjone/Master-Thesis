from utils import rand
import torch
import matplotlib as mpl
from functions.random_functions.rosenbrock_functions import RandomRosenbrock
from functions.random_functions.convex_functions import RandomConvex
from functions.random_functions.himmelblau_functions import RandomHimmelblau
from functions.random_functions.branin_function import RandomBranin
from functions.random_functions.goldsteinprice_function import RandomGoldsteinPrice
from functions.random_functions.hartmann3_function import RandomGP
from functions.random_functions.random_multimodal import RandomMultimodal
import numpy as np 
import matplotlib.pyplot as plt
str_to_class = {"rosenbrock": RandomRosenbrock,
                "convex": RandomConvex,
                "himmelblau": RandomHimmelblau, 
                "branin": RandomBranin,
                "goldsteinprice": RandomGoldsteinPrice,
                "multimodal": RandomMultimodal,
                "gp": RandomGP}


class RandomFunction:

    def __init__(self, config) -> None:

        self.resolution = config.resolution    
        self.batch_size = config.batch_size
        self.dims = config.dims
        self.max_value_range = config.max_value_range
        
        self.function_types = []
        for funcname in config.functions:
            self.function_types.append(str_to_class[funcname.lower()](config))

        self.matrix = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cpu"))
        self.max = torch.zeros((self.batch_size)).to(torch.device("cpu"))
        self.min = torch.zeros((self.batch_size)).to(torch.device("cpu"))
        self.function_classes = torch.zeros((self.batch_size), dtype=torch.long).to(torch.device("cpu"))

        self.reset()


    def reset(self, idx = None):
        if idx is None: idx = torch.arange(start = 0, end = self.batch_size)
        if len(idx.shape) == 0:
            idx = idx.unsqueeze(0)

        #Sample a function class
        func_class = np.random.randint(len(self.function_types))
        func = self.function_types[func_class]
        self.function_classes[idx] = func_class

        #Get the matrix and save it
        matrix = func.get_matrix(idx.shape[0])

        scale_num = rand(self.max_value_range[0], self.max_value_range[1], 1)
        dim = tuple(range(1, self.dims+1))
        if self.dims == 2:
            matrix -= (torch.amin(matrix, dim = dim))[:, None, None]
            matrix = matrix/((torch.amax(matrix, dim = dim)*scale_num)[:, None, None])
        elif self.dims == 3:
            matrix -= (torch.amin(matrix, dim = dim))[:, None, None, None]
            matrix = matrix/((torch.amax(matrix, dim = dim)*scale_num)[:, None, None, None])
        else:
            raise NotImplementedError("Only supports 2D and 3D so far")
        self.matrix[idx] = matrix
        self.max[idx] = torch.amax(matrix, dim = dim)
        self.min[idx] = torch.amin(matrix, dim = dim)
        
        #Do some basic checks
        assert len(self.max.shape) == 1, f"Expected max to have one dimension, found: {self.max.shape}"
        assert torch.min(self.min) == 0, f"Min shouldn't be less than 0, found: {torch.min(self.min)}"
        #assert torch.max(self.max) == 1, f"Max shouldn't be bigger than 1, found: {torch.max(self.max)}"

    def visualize_two_dims(self):
        raise NotImplementedError("Call this on the function type itself")
        matrix = self.matrix
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].cpu().numpy(), cmap='viridis')
            self.function_classes
            ax.set_title('Angle = {:.2f}'.format(self.function_classes[0].params["angles"][i, 0, 0].item()))
        plt.tight_layout()
        plt.show() 

    def visualize_single(self, name, dpi = 600):
        matrix = self.matrix
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"
        

        plt.title(f"Visualization of {name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.imshow(matrix[0].detach().cpu().numpy(), cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
        plt.colorbar()
        
        plt.show()
