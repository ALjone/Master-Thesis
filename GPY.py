import numpy as np
from itertools import product
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, CosineKernel, PolynomialKernel, LinearKernel
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size, kernel, dims):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel(batch_shape=torch.Size([batch_size], ard_num_dims = dims)), batch_shape=torch.Size([batch_size]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class GP:
    def __init__(self, kernels, batch_size, domain, resolution, verbose = 0, learning_rate = 0.1, training_iters = 50, dims = 3) -> None:

        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.kernels = kernels if kernels is not None else [RBFKernel]
        self.verbose = verbose
        self.resolution = resolution
        self.min_, self.max_ = domain[0], domain[1]
        self.device = torch.device("cuda")
        self.dims = dims
        self.batch_size = batch_size

        vectors = [torch.linspace(self.min_, self.max_, self.resolution) for _ in range(dims)]

        # Create a meshgrid of indicesq
        idxs = np.meshgrid(*[torch.arange(len(vec)) for vec in vectors], indexing='ij')

        # Stack the indices and use them to index the original vectors
        points = torch.column_stack([vec[idx.flatten()] for vec, idx in zip(vectors, idxs)])
        # Reshape the result to the desired shape
        self.points = points.repeat_interleave(batch_size, 0).reshape(batch_size, -1, dims).to(self.device)

        test_x = torch.linspace(self.min_, self.max_, self.resolution)
        test_y = torch.linspace(self.min_, self.max_, self.resolution)
        test_xx, test_yy = torch.meshgrid(test_x, test_y, indexing="ij")
        test_xx = test_xx.reshape(-1, 1)
        test_yy = test_yy.reshape(-1, 1)
        self.points = torch.cat([test_xx, test_yy], dim=1).to(torch.device("cuda")).unsqueeze(0).repeat_interleave(self.batch_size, 0)

    def _get_model(self, kernel, x, y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x.shape[0]]))
        model = ExactGPModel(x, y, likelihood, x.shape[0], kernel, self.dims).to(self.device)
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

    def _predict_matrix(self, model: ExactGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood, idx):        

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(self.points[idx])
            observed_pred = likelihood(output)

        #TODO: Scale back?
        self.mean = observed_pred.mean
        self.std = observed_pred.stddev
        _, self.upper_confidence = observed_pred.confidence_region()

        return self.mean.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))), self.std.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims)))
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    batch = 2
    gp = GP(None, batch, (0, 1), 30, dims = 2, verbose=1)
    x = torch.tensor([[0, 0], [0.5, 0.7], [0.3, 0.3]]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cuda"))
    y = torch.tensor([0, 3.4, 1.5]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cuda"))
    #y = torch.rand(3).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cuda"))

    #x = torch.tensor([[0.3, 0.3]]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cuda"))
    #y = torch.tensor([1.5]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cuda"))

    idx = torch.arange(batch).to(torch.device("cuda"))

    mean, interval = gp.get_mean_std(x, y, idx)
    plt.imshow(mean.cpu()[0])
    plt.scatter([0*30, 0.5*30, 0.3*30], [0, 0.7*30, 0.3*30])
    plt.colorbar()
    plt.show()
    plt.close()
    plt.cla()
    plt.imshow(interval.cpu()[0])
    plt.show()
