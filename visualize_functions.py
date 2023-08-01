from utils import load_config
from functions.random_functions.random_functions import RandomFunction
#from random_functions.convex_functions import RandomConvex
#from random_functions.random_functions import RandomFunction
import numpy as np
import torch
#TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
config = load_config("configs\\training_config.yml")
config.batch_size = 10
config.domain = [-1, 1]
config.functions = ["multimodal"]
config.resolution = 100
f = RandomFunction(config)
for i in range(100):
    #f.visualize_two_dims()
    f.visualize_single("a randomly drawn multimodal function")
    f.reset()
