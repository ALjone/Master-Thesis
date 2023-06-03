import torch
from random_functions.rosenbrock_functions import RandomRosenbrock
from random_functions.convex_functions import RandomConvex
import numpy as np 
import matplotlib.pyplot as plt
str_to_class = {"rosenbrock": RandomRosenbrock,
                "convex": RandomConvex}


class RandomFunction:

    def __init__(self, config) -> None:

        self.resolution = config.resolution    
        self.batch_size = config.batch_size
        self.dims = config.dims
        
        self.function_types = []
        for funcname in config.functions:
            self.function_types.append(str_to_class[funcname.lower()](config))

        self.matrix = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cuda"))
        self.max = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        self.min = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        self.function_classes = torch.zeros((self.batch_size), dtype=torch.long).to(torch.device("cuda"))

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
        self.matrix[idx] = matrix
        self.max[idx] = torch.ones((len(idx), )).to(torch.device("cuda"))
        self.min[idx] = torch.amin(matrix, dim = tuple(range(1, self.dims+1)))
        
        #Do some basic checks
        assert len(self.max.shape) == 1, f"Expected max to have one dimension, found: {self.max.shape}"
        assert torch.min(self.min) == 0, f"Min shouldn't be less than 0, found: {torch.min(self.min)}"
        assert torch.max(self.max) == 1, f"Max shouldn't be bigger than 1, found: {torch.max(self.max)}"

    def visualize_two_dims(self):
        matrix = self.matrix
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].cpu().numpy(), cmap='viridis')
        plt.tight_layout()
        plt.show() 

