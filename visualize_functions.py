from utils import load_config
from random_functions.rosenbrock_functions import RandomRosenbrock
from random_functions.convex_functions import RandomConvex
from random_functions.random_functions import RandomFunction
import numpy as np
import torch
#TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
config = load_config("configs\\training_config.yml")
config.batch_size = 10
config.domain = [-1, 1]
f = RandomFunction(config)
f.reset()
for i in range(100):
    f.visualize_two_dims()
    f.reset()
    #f.reset(torch.tensor(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), dtype = torch.long))