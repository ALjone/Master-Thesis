from utils import load_config
from random_functions.rosenbrock_functions import RandomFunction
#TODO: Play with parameters, they ain't good nuff. Copula for å korrelere dimensjonene
config = load_config("configs\\training_config.yml")
#config.batch_size = 2
f = RandomFunction(config)
for i in range(100):
    f.visualize_two_dims()
    f.reset()