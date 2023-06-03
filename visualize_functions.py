from utils import load_config
from random_functions.rosenbrock_functions import RandomRosenbrock
from random_functions.himmelblau_functions import RandomHimmelblau
#from random_functions.convex_functions import RandomConvex
#from random_functions.random_functions import RandomFunction
import numpy as np
import torch
#TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
config = load_config("configs\\training_config.yml")
config.batch_size = 10
config.domain = [-1, 1]
f = RandomHimmelblau(config)
for i in range(100):
    f.visualize_two_dims()
