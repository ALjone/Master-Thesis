import torch
import numpy as np
import matplotlib.pyplot as plt

class RandomGP:
    """
    Implements the sparse spectrum approximation of a GP following the predictive
    entropy search paper.

    Note: This approximation assumes that we use a GP with squared exponential kernel.
    """

    def __init__(self, config, noise_var=1.0, length_scale=1.0, signal_var=1.0, n_features=100, kernel="RBF"):
        self.rng = np.random.RandomState()

        self.dims = config.dims
        self.noise_var = noise_var
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.n_features = n_features
        self.kernel = kernel
        self.batch_size = config.batch_size
        assert kernel == "RBF" or kernel == "Matern32" or kernel == "Matern52"
        self.phi = self._compute_phi()
        self.jitter = 1e-10

        self.range = [-5, 5]
        self.resolution = config.resolution
        x = np.linspace(self.range[0], self.range[1], self.resolution)
        X, Y = np.meshgrid(x, x)
        X = np.repeat(X[None, :, :], self.batch_size, axis=0)
        Y = np.repeat(Y[None, :, :], self.batch_size, axis=0)
        self.x = X #torch.tensor(X).to(torch.device("cuda"))
        self.y = Y #torch.tensor(Y).to(torch.device("cuda"))

        self.theta_mu = None
        self.theta_var = None
        self.theta_samples = None

    def _compute_phi(self):
        """
        Compute random features.
        """
        if self.kernel == "RBF":
            w = self.rng.randn(self.n_features, self.dims) / self.length_scale
        elif self.kernel == "Matern32":
            w = self.rng.standard_t(3, (self.n_features, self.dims)) / self.length_scale
        elif self.kernel == "Matern52":
            w = self.rng.standard_t(5, (self.n_features, self.dims)) / self.length_scale
        b = self.rng.uniform(0, 2 * np.pi, size=self.n_features)
        return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(x @ w.T + b)

    def get_matrix(self, size):
        if self.theta_samples is None:
            # Generate handle to n_samples function samples that can be evaluated at x.
            var = self.jitter * np.eye(self.n_features)
            chol = np.linalg.cholesky(var)
            self.theta_samples = chol @ self.rng.randn(self.n_features, size)

        x_grid = np.array([self.x, self.y])
        x_grid = x_grid.reshape(self.dims, -1).T

        phi_x = self.phi(x_grid)
        sampled_functions = self.theta_samples.T @ phi_x.T
        print(sampled_functions.shape)
        sampled_functions_grid = sampled_functions.reshape(size, self.resolution, self.resolution)
        return sampled_functions_grid

    def visualize_two_dims(self):
        matrix = self.get_matrix(10)
        assert self.dims == 2, f"Can only visualize 2 dims, found: {self.dims}"

        fig = plt.figure(figsize=(12, 8))
        for i in range(min(10, self.batch_size)):
            ax = fig.add_subplot(2, 5, i + 1)
            ax.imshow(matrix[i].detach().cpu().numpy(), cmap='viridis')
            #ax.set_title('A = {:.2f}, B = {:.2f}, angle = {:.2f}'.format(self.params["a"][i].item(), self.params["b"][i].item(), self.params["angles"][i, 0, 0].item()))
        plt.tight_layout()
        plt.show() 