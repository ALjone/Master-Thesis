from typing import Any, List
import torch
import numpy as np 
import matplotlib.pyplot as plt
from utils import rand

class Linear():
    def __init__(self, coefficient_range: List[int], constant_range: List[int], dims: int) -> None:
        """Constants is a list of length n+1, where n is the number of dimensions. The last element is the constant factor"""
        self.coefficient_min = coefficient_range[0]
        self.coefficient_max = coefficient_range[1]
        self.constant_min = constant_range[0]
        self.constant_max = constant_range[1]
        self.dims = dims
    
    def __call__(self, points) -> Any:
        result = torch.zeros_like(points[..., 0], device=torch.device("cuda"))  # Initialize result tensor

        for i in range(self.dims):
            constant = rand(self.coefficient_min, self.coefficient_max, result.shape[0])
            result += constant * points[..., i]

        return result+rand(self.constant_min, self.constant_max, result.shape[0])

#Thanks to ChatGPT4
class Polynomial():
    def __init__(self, exponent_range: List[int], constant_range: List[int]) -> None:
        """power is a list of length n+1, where n is the number of dimensions. The last element is the constant factor"""
        self.exponent_min = exponent_range[0]
        self.exponent_max = exponent_range[1]
        self.constant_min = constant_range[0]
        self.constant_max = constant_range[1]
    
    def __call__(self, points) -> Any:

        result = torch.zeros_like(points[..., 0], device=torch.device("cuda"))  # Initialize result tensor

        for i in self.dims:
            constant = rand(self.exponent_min, self.exponent_max, result.shape[0])
            result += points[..., i]**constant

        return result+self.exponent_range[-1]


str_to_class = {"linear": Linear,
                "polynomial": Polynomial}

class TimeFunction:

    def __init__(self, config) -> None:

        self.resolution = config.resolution    
        self.batch_size = config.batch_size
        self.dims = config.dims
        
        self.function_types = config.time_functions
        #for funcname in :
        self.function_types_probabilities = config.time_function_probabilities

        self.time_matrix = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cuda"))
        self.max = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        self.min = torch.zeros((self.batch_size)).to(torch.device("cuda"))
        self.time_classes = torch.zeros((self.batch_size), dtype=torch.long).to(torch.device("cuda"))

        x = torch.linspace(0, config.domain[0]-config.domain[1], self.resolution, device=torch.device("cuda"))  # Generate equally spaced values between -1 and 1
        grids = torch.meshgrid(*([x] * config.dims))  # Create grids for each dimension
        self.points = torch.stack(grids, dim=-1)  # Stack grids along the last axis to get points
        print(self.points.shape)
        print(grids.shape)
        shape = [self.batch_size] + list(self.points.shape)
        self.x = torch.repeat_interleave(self.points, self.batch_size).reshape(shape).to(torch.device("cuda"))
        print(self.x.shape)
        exit()
        lin = Linear(config.linear_range, config.constant, config.dims)
        print(lin(self.points).shape)

        self.reset()


    def reset(self, idx = None):
        if idx is None: idx = torch.arange(start = 0, end = self.batch_size)
        if len(idx.shape) == 0:
            idx = idx.unsqueeze(0)

        #Sample a function class
        func_class = np.random.randint(len(self.function_types))
        func = self.function_types[func_class]
        self.time_classes[idx] = func_class

        #Get the matrix and save it
        matrix = func.get_matrix(idx.shape[0])
        self.time_matrix[idx] = matrix
        self.max[idx] = torch.ones((len(idx), )).to(torch.device("cuda"))
        self.min[idx] = torch.amin(matrix, dim = tuple(range(1, self.dims+1)))
        
        #Do some basic checks
        assert len(self.max.shape) == 1, f"Expected max to have one dimension, found: {self.max.shape}"
        assert torch.min(self.min) == 0, f"Min shouldn't be less than 0, found: {torch.min(self.min)}"
        assert torch.max(self.max) == 1, f"Max shouldn't be bigger than 1, found: {torch.max(self.max)}"

    def visualize_two_dims(self):
        matrix = self.time_matrix
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].cpu().numpy(), cmap='viridis')
        plt.tight_layout()
        plt.show() 

