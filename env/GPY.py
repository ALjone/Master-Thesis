import numpy as np
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size, kernel, mixtures = None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([batch_size]))
        if mixtures is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims = train_x.shape[-1], batch_shape = torch.Size([batch_size]), num_mixtures = mixtures), batch_shape=torch.Size([batch_size]))
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims = train_x.shape[-1], batch_shape = torch.Size([batch_size])), batch_shape=torch.Size([batch_size]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, batch_size, kernels, operation, dims, device):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(1), batch_shape=torch.Size([batch_size])).to(device)
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([batch_size]))
        if operation == 'add':
            combined_kernel = gpytorch.kernels.AdditiveKernel(*kernels)
        else:  # operation == 'mul'
            combined_kernel = gpytorch.kernels.ProductKernel(*kernels)
        self.covar_module = gpytorch.kernels.ScaleKernel(combined_kernel, batch_shape=torch.Size([batch_size]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class GP:
    def __init__(self, config) -> None:
        self.noise = config.noise
        self.training_iters = config.GP_training_iters
        self.learning_rate = config.GP_learning_rate
        self.kernel = config.kernel
        self.operation = config.operation
        self.verbose = config.verbose
        self.resolution = config.resolution
        self.min_, self.max_ = config.domain[0], config.domain[1]
        self.device = torch.device("cpu")
        self.dims = config.dims
        self.batch_size = config.batch_size
        self.approximate = config.approximate
        self.mixtures = config.mixtures
        self.kernel_type = config.kernel_name
        self.length_scales = torch.zeros(self.batch_size)
        


        self.mean = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cpu"))
        self.std = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cpu"))
        self.EI = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cpu"))
        self.UCB = torch.zeros((self.batch_size, ) + tuple(self.resolution for _ in range(self.dims))).to(torch.device("cpu"))
        self.biggest = torch.zeros((self.batch_size, ))

        if self.dims == 2:
            test_x = torch.linspace(self.min_, self.max_, self.resolution)
            test_y = torch.linspace(self.min_, self.max_, self.resolution)
            test_xx, test_yy = torch.meshgrid(test_x, test_y, indexing="ij")
            test_xx = test_xx.reshape(-1, 1)
            test_yy = test_yy.reshape(-1, 1)
            self.points = torch.cat([test_xx, test_yy], dim=1).to(torch.device("cpu")).unsqueeze(0).repeat_interleave(self.batch_size, 0)
        elif self.dims == 3:
            test_x = torch.linspace(self.min_, self.max_, self.resolution)
            test_y = torch.linspace(self.min_, self.max_, self.resolution)
            test_z = torch.linspace(self.min_, self.max_, self.resolution)
            test_xx, test_yy, test_zz = torch.meshgrid(test_x, test_y, test_z, indexing="ij")
            test_xx = test_xx.reshape(-1, 1)
            test_yy = test_yy.reshape(-1, 1)
            test_zz = test_zz.reshape(-1, 1)
            self.points = torch.cat([test_xx, test_yy, test_zz], dim=1).to(torch.device("cpu")).unsqueeze(0).repeat_interleave(self.batch_size, 0)
        else:
            raise NotImplementedError("Only dims = 2 or dims = 3 is currently implemented because ")

    def _get_model(self, x, y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x.shape[0]])).to(self.device)

        if self.approximate:
            model = ApproximateGPModel(x, x.shape[0], self.kernel, self.dims, self.device).to(self.device)
        else:
            model = ExactGPModel(x, y, likelihood, x.shape[0], self.kernel, self.mixtures if self.kernel_type == "sm" else None).to(self.device)

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
        model, likelihood, _ = self._get_model(x, y)
        #self.length_scales[idx] = model.covar_module.base_kernel.lengthscale
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
        self.mean[idx] = observed_pred.mean.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims)))
        self.std[idx] = observed_pred.stddev.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims)))
        _, UCB = observed_pred.confidence_region()
        self.UCB[idx] = UCB.reshape((-1, ) + tuple(self.resolution for _ in range(self.dims)))

        mean = self.mean[idx].cpu()
        std = self.std[idx].cpu()

        with torch.no_grad():
            # Compute Z scores
            biggest = torch.amax(self.y, dim=(1)).cpu()
            for _ in range(self.dims):
                biggest = biggest.unsqueeze(1)
            e = 0.001 #TODO: Hyperparameter tune e
            Z = (mean - biggest - e) / std

            EI = ((mean - biggest - e) * norm.cdf(Z) + std * norm.pdf(Z)).to(torch.device("cpu")).to(torch.float)
            if self.dims == 2:
                min_values = torch.amin(EI, dim=tuple(i for i in range(1, self.dims+1))).unsqueeze(1).unsqueeze(1)
                max_values = torch.amax(EI, dim=tuple(i for i in range(1, self.dims+1))).unsqueeze(1).unsqueeze(1)
            elif self.dims == 3:
                min_values = torch.amin(EI, dim=tuple(i for i in range(1, self.dims+1))).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                max_values = torch.amax(EI, dim=tuple(i for i in range(1, self.dims+1))).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                raise NotImplementedError("Fix unsqueeze to work in n-dims")
            #diff = (max_values - min_values)

            #diff[diff == 0] = 1
            # Normalize the tensor within each batch
            #normalized_EI = (EI - min_values) / (diff + 1e-5)

            # Compute expected improvement for all points in batch
        self.EI[idx] = EI

        if self.mean.isnan().sum() > 0:
            print("Found NaNs in mean!")
            print("Found", self.mean.isnan().sum().item(), "NaNs")
        if self.std.isnan().sum() > 0:
            print("Found NaNs in std!")
            print("Found", self.std.isnan().sum().item(), "NaNs")
        if self.EI.isnan().sum() > 0:
            print("Found NaNs in EI!")
            print("Found", self.EI.isnan().sum().item(), "NaNs")

        return self.mean[idx], self.std[idx], self.EI[idx], self.UCB[idx]

    def get_next_point(self, return_idx = True):
        EI = self.EI.cpu().numpy()
        best_indices = []
        for i in range(self.batch_size):
            best_index = np.unravel_index(np.argmax(EI[i], axis=None), EI[i].shape)
            best_indices.append(best_index)

        idx = torch.tensor(best_indices).to(torch.device("cpu")) 
        return idx if return_idx else ((idx+1)-(self.resolution//2))/(self.resolution//2)
    
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
    x = torch.tensor([[0, 0], [0.5, 0.7], [0.3, 0.3]]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cpu"))
    y = torch.tensor([0, 3.4, 1.5]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cpu"))
    #y = torch.rand(3).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cpu"))

    #x = torch.tensor([[0.3, 0.3]]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cpu"))
    #y = torch.tensor([1.5]).unsqueeze(0).repeat_interleave(batch, 0).to(torch.device("cpu"))

    idx = torch.arange(batch).to(torch.device("cpu"))

    mean, interval = gp.get_mean_std(x, y, idx)
    plt.imshow(mean.cpu()[0])
    plt.scatter([0*30, 0.5*30, 0.3*30], [0, 0.7*30, 0.3*30])
    plt.colorbar()
    plt.show()
    plt.close()
    plt.cla()
    plt.imshow(interval.cpu()[0])
    plt.show()
