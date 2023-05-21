import numpy as np
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel#, CosineKernel, PolynomialKernel, LinearKernel
import torch
from matplotlib import pyplot as plt


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
    def __init__(self, kernels, batch_size, domain, resolution, verbose = 0, learning_rate = 0.1, training_iters = 200, dims = 3, approximate = False, noise = None) -> None:
        self.noise = noise
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.kernels = kernels if kernels is not None else [RBFKernel]
        self.verbose = verbose
        self.resolution = resolution
        self.min_, self.max_ = domain[0], domain[1]
        self.device = torch.device("cuda")
        self.dims = dims
        self.batch_size = batch_size
        self.approximate = approximate
        if dims == 2:
            test_x = torch.linspace(self.min_, self.max_, self.resolution)
            test_y = torch.linspace(self.min_, self.max_, self.resolution)
            test_xx, test_yy = torch.meshgrid(test_x, test_y, indexing="ij")
            test_xx = test_xx.reshape(-1, 1)
            test_yy = test_yy.reshape(-1, 1)
            self.points = torch.cat([test_xx, test_yy], dim=1).to(torch.device("cuda")).unsqueeze(0).repeat_interleave(self.batch_size, 0)
        elif dims == 3:
            test_x = torch.linspace(self.min_, self.max_, self.resolution)
            test_y = torch.linspace(self.min_, self.max_, self.resolution)
            test_z = torch.linspace(self.min_, self.max_, self.resolution)
            test_xx, test_yy, test_zz = torch.meshgrid(test_x, test_y, test_z, indexing="ij")
            test_xx = test_xx.reshape(-1, 1)
            test_yy = test_yy.reshape(-1, 1)
            test_zz = test_zz.reshape(-1, 1)
            self.points = torch.cat([test_xx, test_yy, test_zz], dim=1).to(torch.device("cuda")).unsqueeze(0).repeat_interleave(self.batch_size, 0)
        else:
            raise NotImplementedError("Only dims = 2 or dims = 3 is currently implemented because ")

    def _get_model(self, kernel, x, y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x.shape[0]])).to(self.device)

        if self.approximate:
            model = ApproximateGPModel(x, x.shape[0], kernel, self.dims, self.device).to(self.device)
        else:
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
            model, likelihood, loss = self._get_model(kernel, x[idx], y[idx])
            if best[0] > loss:
                best = (loss, model, likelihood)
            if self.verbose:
                print("Kernel:", kernel, "Loss:", loss)
        
        #Save best GPR
        model = best[1]
        likelihood = best[2]

        self.x = x
        self.y = y

        return self._predict_matrix(model, likelihood, idx)

    def _predict_matrix(self, model: ApproximateGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood, idx):        

        model.eval()
        likelihood.eval()
        if self.noise is not None:
            likelihood.noise = self.noise

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(self.points[idx])
            observed_pred = likelihood(output)

        #TODO: Scale back?
        self.mean = observed_pred.mean
        self.std = observed_pred.stddev
        self.lower_confidence, self.upper_confidence = observed_pred.confidence_region()

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
    

    def get_next_point(self, biggest, e = 0.001):
        from scipy.stats import norm
        mean = self.mean.reshape((-1,) + tuple(self.resolution for _ in range(self.dims))).cpu()
        std = self.std.reshape((-1,) + tuple(self.resolution for _ in range(self.dims))).cpu()

        # Compute Z scores
        Z = (mean - biggest - e) / std

        # Compute expected improvement for all points in batch
        improvement = (mean - biggest - e) * norm.cdf(Z) + std * norm.pdf(Z)

        # Find the indices of the best points for each element in the batch
        best_indices = []
        print(improvement.shape, self.mean.shape, self.std.shape)
        for i in range(self.batch_size):
            best_index = np.unravel_index(np.argmax(improvement[i], axis=None), improvement[i].shape)
            best_indices.append(best_index)

        return torch.tensor(best_indices).to(torch.device("cuda")) 

    def render(self, show = False):
        plt.close()
        plt.cla()
        assert self.dims == 2, f"Only support 2 dims atm, found: {self.dims}"
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter((self.x[0, :, 0].cpu())*self.resolution, (self.x[0, :, 1].cpu())*self.resolution, c = "red")
        axs[1].scatter((self.x[0, :, 0].cpu())*self.resolution, (self.x[0, :, 1].cpu())*self.resolution, c = "red")
        img = axs[0].imshow(self.mean.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))).cpu()[0].T)
        plt.colorbar(img, ax = axs[0])
        img = axs[1].imshow(self.std.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims))).cpu()[0].T)
        plt.colorbar(img, ax = axs[1])
        axs[0].set_title("Mean")
        axs[1].set_title("Std")
        if show:
            plt.show()
            data = None
        else:
            fig.canvas.draw()
            arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        plt.cla()
        return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    batch = 2
    gp = GP(None, batch, (0, 1), 30, dims = 2, verbose=1, training_iters=200, approximate=False)
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
