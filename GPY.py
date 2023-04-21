import numpy as np
from itertools import product
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel(batch_shape=torch.Size([batch_size])), batch_shape=torch.Size([batch_size]))
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, kernels, batch_size, domain, resolution, verbose = 0, learning_rate = 0.1, training_iters = 100) -> None:

        #TODO: REWRITE TO WORK FROM 0-1 OR SOMETHING
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.kernels = kernels if kernels is not None else [MaternKernel, RBFKernel]
        self.verbose = verbose
        self.resolution = resolution
        self.min_, self.max_ = domain[0], domain[1]
        self.device = torch.device("cuda")

        #TODO: This should be self.points, and be batch_size x self.resolution^3 x 3
        self.matrix = torch.einsum('i,j,k->ijk', torch.linspace(self.min_, self.max_, self.resolution), torch.linspace(self.min_, self.max_, self.resolution), torch.linspace(self.min_, self.max_, self.resolution)).to(self.device)
        vector1 = torch.linspace(self.min_, self.max_, self.resolution)
        vector2 = torch.linspace(self.min_, self.max_, self.resolution)
        vector3 = torch.linspace(self.min_, self.max_, self.resolution)

        # Create a meshgrid of indices
        idx1, idx2, idx3 = np.meshgrid(torch.arange(len(vector1)),
                                    torch.arange(len(vector2)),
                                    torch.arange(len(vector3)),
                                    indexing='ij')

        # Stack the indices and use them to index the original vectors
        points = torch.column_stack((vector1[idx1.flatten()],
                                vector2[idx2.flatten()],
                                vector3[idx3.flatten()]))

        # Reshape the result to the desired shape
        self.points = points.repeat_interleave(batch_size, 0).reshape(batch_size, -1, 3).to(self.device)


    def _get_model(self, kernel, x, y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x.shape[0]]))
        model = ExactGPModel(x, y, likelihood, x.shape[0], kernel).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        for _ in range(self.training_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x)
            # Calc loss and backprop gradients
            loss = -mll(output, y).sum()
            loss.backward()

            optimizer.step()
        return model, likelihood, loss.item()

    def get_mean_std(self, x, y, idx):
        if len(x.shape) == 5:
            x = x.squeeze().unsqueeze(0)
            y = y.squeeze().unsqueeze(0)
        #NOTE: Assumes scaled input
        best = (np.inf, None, None)
        for kernel in self.kernels:
            model, likelihood, loss = self._get_model(kernel, x, y)
            if best[0] > loss:
                best = (loss, model, likelihood)
            if self.verbose:
                print("Kernel:", kernel, "Loss:", loss)
        
        #Save best GPR
        model = best[1]
        likelihood = best[2]

        return self._predict_matrix(model, likelihood, idx)

    def _predict_matrix(self, model, likelihood, idx):        
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(self.points[idx])
            observed_pred = likelihood(output)
        
        #TODO: Scale back?
        self.mean = observed_pred.mean
        self.std = observed_pred.stddev

        return self.mean.reshape(-1, self.resolution, self.resolution, self.resolution), self.std.reshape(-1, self.resolution, self.resolution, self.resolution)
