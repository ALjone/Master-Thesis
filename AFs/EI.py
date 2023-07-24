with torch.no_grad():
    # Compute Z scores
    biggest = torch.amax(self.y, dim=(1)).cpu()
    for _ in range(self.dims):
        biggest = biggest.unsqueeze(1)
    e = 0.001 #TODO: Hyperparameter tune e
    Z = (mean - biggest - e) / std

    EI = ((mean - biggest - e) * norm.cdf(Z) + std * norm.pdf(Z)).to(torch.device("cuda")).to(torch.float)
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