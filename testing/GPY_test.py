import time
import torch
import gpytorch
batch_size = 1000

for batch_size in [1, 10, 100, 1000, 10000]: 
    train_x = torch.rand((batch_size, 50, 3))
    train_y = torch.mean(train_x, dim=2)

    train_x, train_y = train_x.to(torch.device("cpu")), train_y.to(torch.device("cpu"))


    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size])), batch_shape=torch.Size([batch_size]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([batch_size]))
    model = ExactGPModel(train_x, train_y, likelihood).to(torch.device("cpu"))

    train_iters = 200


    test_x = torch.rand((batch_size, 100, 3)).to(torch.device("cpu"))
    test_y = torch.mean(test_x, dim=2).to(torch.device("cpu"))

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(test_x)
        observed_pred = likelihood(output)

    pre_train_loss = torch.nn.MSELoss()(observed_pred.mean, test_y)

    # this is for running the notebook in our testing framework

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    s = time.time()
    for i in range(train_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y).sum()
        loss.backward()
        """print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, train_iters, loss.item()/batch_size,
            model.covar_module.base_kernel.lengthscale.mean().item(),
            model.likelihood.noise.mean().item()
        ))"""
        optimizer.step()

    print(f"Batch size: {batch_size} Training took {round(time.time()-s, 2)} seconds, for an average of {round((time.time()-s)/(batch_size), 5)} seconds per GP")

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(test_x)
        observed_pred = likelihood(output)

    print("Test shape:", test_x.shape, "Output shape:", observed_pred.mean.shape)
    loss = torch.nn.MSELoss()(observed_pred.mean, test_y)
    #print("Pre train loss:", pre_train_loss.item(), "Post train loss:", loss.item())