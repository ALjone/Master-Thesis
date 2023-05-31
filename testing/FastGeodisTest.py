import FastGeodis
from matplotlib import pyplot as plt
import torch

a = torch.rand((40, 40))

a[a > 0.9] = 1
a[a<1] = 0

print(a)