import torch
from utils import load_config
from functions.time_functions.time_functions import TimeFunction
config = load_config("configs\\training_config.yml")

TimeFunction(config)
"""#TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
config.batch_size = 2048*4
config.domain = [-1, 1]
f = RandomRosenbrock(config)
f.reset()
import matplotlib.pyplot as plt

matrix = f.matrix

plt.imshow(torch.mean(matrix, dim = 0).cpu().numpy())
plt.show()"""