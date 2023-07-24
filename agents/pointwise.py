import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch import nn
import torch

class Agent(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, layer_size, dims = 2, verbose = True):
        super().__init__()
        self.dims = dims
        self.n_features = observation_space.shape[1]
        self.actor = nn.Sequential( nn.Linear(self.n_features, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, 1),
                                    nn.LeakyReLU())
        
        self.critic = nn.Sequential( nn.Linear(2, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, layer_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(layer_size, 1),
                                    nn.LeakyReLU())
        if verbose:
            print("Running with", self.count_parameters(), "parameters")
    #Thanks to ChatGPT
    def get_flattened_index(self, coords, resolution):
        """Convert coordinates to their corresponding flattened index."""
        dims = coords.size(1)

        # Calculate the stride for each dimension
        strides = torch.tensor([resolution ** (dims - i - 1) for i in range(dims)]).unsqueeze(0).to(coords.device)

        # Multiply the coordinates by the strides and sum along the dimensions to obtain the flattened index
        idx = torch.sum(coords * strides, dim=1, keepdim=True)

        return idx
    
    #Thanks to ChatGPT
    def get_coords(self, idx, dims, resolution):
        """Convert a flattened index to its corresponding coordinates."""
        coords = torch.zeros(idx.shape[0], dims, dtype=torch.int64, device=idx.device)

        if dims == 2:
            coords[:, 1] = idx % resolution
            coords[:, 0] = idx // resolution

        elif dims == 3:
            coords[:, 2] = idx % resolution
            idx = idx // resolution
            coords[:, 1] = idx % resolution
            coords[:, 0] = idx // resolution

        else:
            raise ValueError("dims should be 2 or 3")

        return coords


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.squeeze()

        # Create a permutation that moves the last dimension to the front and keeps the rest in order.
        perm = (0, ) + tuple(range(2, len(observations.shape))) + (1, )
        reshaped_observations = observations.permute(perm).reshape(-1, self.n_features)

        action = self.actor(reshaped_observations)

        # Select the last two elements from the final dimension and the first element from all other dimensions.
        global_dims = tuple([slice(None), slice(self.n_features-2, self.n_features, 1)] + [0]*(self.dims))
        critic = self.critic(observations[global_dims])

        action_view_shape = tuple([-1] + [observations.shape[-1]] * (self.dims))
        return action.view(*action_view_shape), critic

    def get_value(self, x):
        _, value = self(x)
        return value

    def get_action_and_value(self, img, action=None, testing = False):
        actor_logits, critic_output = self(img)

        logits = actor_logits
        categorical = Categorical(logits = logits.flatten(1))
        if action is None:
            if not testing:
                action = categorical.sample() 
            else:
                action = categorical.mode
        else: 
            action = self.get_flattened_index(action, img.shape[-1]).squeeze() #action[:, 0]*actor_logits.size(2)+action[:, 1]#Turn it from (x, y) to just one indx
        log_prob = categorical.log_prob(action)
        #shape = (img.shape[0], ) + tuple(s for s in img.shape[2:])
        indices_tensor = self.get_coords(action, self.dims, img.shape[-1])
        #get_coords(self, idx, dims, resolution)

        return indices_tensor, log_prob, categorical.entropy(), critic_output
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def visualize_dist(self, img, t):
        plt.cla()
        plt.close()
        with torch.no_grad():
            prob, _ = self(img, t)
        plt.imshow(prob.cpu().numpy())
        plt.show()

    def get_2d_logits_and_prob(self, img):
        with torch.no_grad():
            logits, _ = self(img)
        print(logits.shape)
        flatten_logits = logits.flatten(1)
        print(flatten_logits.shape)
        flattened_probs = torch.nn.Softmax(dim=1)(flatten_logits)
        probs = flattened_probs.reshape(logits.shape)

        return logits.cpu().numpy(), probs.cpu().numpy()
    
    def visualize_positional_weighting(self):
        return