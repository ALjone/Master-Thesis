import torch
import matplotlib.pyplot as plt
import numpy as np

# Parameters for the 2D normal distribution
mu = torch.zeros(2)
cov = torch.eye(2)

# Create a Normal distribution with these parameters
normal_dist = torch.distributions.MultivariateNormal(mu, cov)

# Create a grid of points at which to evaluate the pdf
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
xy = np.column_stack([X.flat, Y.flat])

# Evaluate the pdf at these points
Z = normal_dist.log_prob(torch.tensor(np.tanh(xy))).exp()
Z = Z.reshape(X.shape)

# Plot the pdf
plt.figure(figsize=(5,5))
plt.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.title('2D Normal Distribution PDF')
plt.show()