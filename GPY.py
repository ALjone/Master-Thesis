import numpy as np
from itertools import product
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, CosineKernel, PolynomialKernel, LinearKernel
from gpytorch.priors import Prior
import torch

class RBFKernelWithPrior(gpytorch.kernels.RBFKernel):
    def __init__(self, **kwargs):
        super(RBFKernelWithPrior, self).__init__(**kwargs)
        
        # Create a length scale parameter with a prior of 1
        self.register_parameter(
            name="lengthscale_prior",
            parameter=torch.nn.Parameter(torch.tensor(1.0)),
            prior=gpytorch.priors.NormalPrior(0, 1)
        )
        
    def forward(self, x1, x2, **params):
        # Retrieve the length scale value from the prior parameter
        lengthscale = self.lengthscale_prior.item()
        
        # Use the length scale value in the kernel computation
        return super(RBFKernelWithPrior, self).forward(x1, x2, **params) * lengthscale ** 2

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size, kernel, dims):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel(batch_shape=torch.Size([batch_size], ard_num_dims = dims)), batch_shape=torch.Size([batch_size]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, batch_size, kernel, dims, device):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(1), batch_shape=torch.Size([batch_size])).to(device)
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel(batch_shape=torch.Size([batch_size], ard_num_dims = dims)), batch_shape=torch.Size([batch_size]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class GP:
    def __init__(self, kernels, batch_size, domain, resolution, verbose = 0, learning_rate = 0.1, training_iters = 200, dims = 3, approximate = False) -> None:

        #TODO: REWRITE TO WORK FROM 0-1 OR SOMETHING

        #TODO: REALLY NEED TO MAKE SURE THAT ACTUALLY JUST REPEATING ONE POINT WORKS
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.kernels = kernels if kernels is not None else [RBFKernelWithPrior]
        self.verbose = verbose
        self.resolution = resolution
        self.min_, self.max_ = domain[0], domain[1]
        self.device = torch.device("cuda")
        self.dims = dims
        self.batch_size = batch_size
        self.approximate = approximate

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
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x.shape[0]])).to(self.device)
        #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.zeros(x.shape[1:]), batch_shape=torch.Size([x.shape[0]])).to(self.device)
        if self.approximate:
            model = ApproximateGPModel(x, x.shape[0], kernel, self.dims, self.device).to(self.device)
        else:
            #model = ExactGPModel(x, y, likelihood, x.shape[0], kernel, self.dims).to(self.device)
            model = gpytorch.models.ExactGP(x, y, kernel).to(self.device)
            
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

    def _predict_matrix(self, model: ApproximateGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood, idx):        

        model.eval()
        likelihood.eval()
        #likelihood.noise_covar.noise = 0.0001
        #likelihood.noise = 0.0001

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(self.points[idx])
            observed_pred = likelihood(output)

        #TODO: Scale back?
        self.mean = observed_pred.mean
        self.std = observed_pred.stddev
        _, self.upper_confidence = observed_pred.confidence_region()

        return self.mean.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))), self.std.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims)))
    
    def get_next_point(self, acquisition, biggest):
        best_point = None
        best_ei = -np.inf
        mean = self.mean.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))).cpu()
        std = self.std.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))).cpu()
        for i in range(mean.shape[1]):
            for j in range(mean.shape[2]):
                ei = acquisition(mean[0, i,j], std[0, i,j], biggest)
                if ei > best_ei:
                    best_point = (i,j)
                    best_ei = ei
        return best_point
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    batch = 2
    gp = GP(None, batch, (0, 1), 30, dims = 2, verbose=1, training_iters=200, approximate=True)
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
