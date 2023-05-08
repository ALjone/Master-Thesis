import time
import torch
import gpytorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

x = torch.tensor([[0, 0], [0.5, 0.7], [0.3, 0.3]]).to(torch.device("cpu"))
y = torch.tensor([0, 3.4, 1.5]).to(torch.device("cpu"))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x, y, likelihood).to(torch.device("cpu"))

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

s = time.time()
for i in range(1000):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x)
    # Calc loss and backprop gradients
    loss = -mll(output, y).sum()
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, 100, loss.item(),
        model.covar_module.base_kernel.lengthscale.mean().item(),
        model.likelihood.noise.mean().item()
    ))
    optimizer.step()

test_x = torch.linspace(0, 1, 30)
test_y = torch.linspace(0, 1, 30)
test_xx, test_yy = torch.meshgrid(test_x, test_y, indexing="ij")
test_xx = test_xx.reshape(-1, 1)
test_yy = test_yy.reshape(-1, 1)
points = torch.cat([test_xx, test_yy], dim=1).to(torch.device("cpu"))
model.eval()
print(model.covar_module)
likelihood.eval()
model.likelihood.noise = 1.01e-04
for p in model.likelihood.named_parameters():
    print(p)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    output = model(points)
    observed_pred = likelihood(output)

#TODO: Scale back?
mean = observed_pred.mean.cpu().reshape(30, 30)
std = observed_pred.stddev.cpu().reshape(30, 30)
_, upper_confidence = observed_pred.confidence_region()
upper_confidence = upper_confidence.cpu().reshape(30, 30)
import matplotlib.pyplot as plt

plt.imshow(mean)
plt.show()