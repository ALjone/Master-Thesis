import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Normal, Categorical
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from torch import nn
from torchvision.ops import SqueezeExcitation
import torch
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, n_channels: int, rescale_input: bool, reduction: int = 16):
        super(SELayer, self).__init__()
        self.rescale_input = rescale_input
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        if self.rescale_input:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)
        else:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, reduction: int = 16, activation = nn.LeakyReLU(), conv = nn.Conv2d):
        """A copy of the conv block from last years winner. Reduction is how many times to reduce the size in the SE"""
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.conv0 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
        self.SE = SELayer(out_channels, True, reduction)

        self.activation = activation
        
        if in_channels != out_channels:
            self.change_channels = conv(in_channels, out_channels, 1)
        else:
            self.change_channels = lambda x: x

    def forward(self, x):
        pre = x
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.SE(x))

        x = x + self.change_channels(pre)

        return self.activation(x)

class Agent(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, dims = 3, use_batch_norm = False):
        super().__init__()
        self.dims = dims
        if dims == 3:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dims == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else: 
            raise ValueError("Only dims = 2 or dims = 3 is currently supported")
        
        layer = ConvBlock
        
        blocks = []

        #Make shared part
        blocks.append(conv(observation_space.shape[1]+1, 8, kernel_size=5, padding = 2))
        blocks.append(nn.LeakyReLU())
        for _ in range(5):
            blocks.append(layer(8, 8, kernel_size=5))
            if use_batch_norm:
                blocks.append(bn(8))


        self.conv = nn.Sequential(*blocks)
        
        self.critic_output = nn.Sequential(nn.Linear(8, 64),
                                           nn.LeakyReLU(),
                                           nn.Linear(64, 32),
                                           nn.LeakyReLU(),
                                           nn.Linear(32, 1))
        
        self.positional_weighting = nn.Parameter(torch.ones(observation_space.shape[2:]).unsqueeze(0))

        print("Running with", self.count_parameters(), "parameters")


    def forward(self, observations: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observations = observations.squeeze()
        for i in range(len(observations.shape)-1):
            time = time.unsqueeze(1)

        global_features = torch.ones((observations.shape[0], 1) + tuple(observations.shape[-1] for _ in range(len(observations.shape)-2)), device=torch.device("cpu"))*time #Make it fit the observation, with 1 channel, to stack
        if observations.isnan().any():
            print("Found NaN in observation!!!")
            print(observations.isnan().sum().item(), "NaNs founds")

        x = torch.cat((observations, global_features), dim = 1)
        x = self.conv(x)

        action = observations[:, 2]*self.positional_weighting


        critic = self.critic_output(nn.AvgPool2d(x.shape[-2])(x).flatten(1))

        return action.squeeze(), critic


    def get_value(self, x, time):
        _, value = self(x, time)
        return value

    def get_action_and_value(self, img, time, action=None, testing = False):
        #TODO!!! We take in action, but find logprob of x_t?
        actor_logits, critic_output = self(img, time)

        #normalized_probs = self.get_normalized_probs(actor_logits, prob_threshold=0.8)
        logits = actor_logits
        categorical = Categorical(logits = logits.flatten(1))
        if action is None:
            if not testing:
                action = categorical.sample()
            else:
                action = categorical.mode
        else: 
            action = action[:, 0]*actor_logits.size(2)+action[:, 1]#Turn it from (x, y) to just one indx
        log_prob = categorical.log_prob(action)

        # Convert the flattened indices to row and column indices
        row_indices = action // actor_logits.size(2)
        col_indices = action % actor_logits.size(2)

        # Stack the row and column indices to create the final tensor of shape (batch, 2)
        indices_tensor = torch.stack((row_indices, col_indices), dim=1)

        #TODO: Should this be / then % or opposite?
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

    def get_2d_logits_and_prob(self, img, t):
        with torch.no_grad():
            logits, _ = self(img, t)

        flatten_logits = logits.flatten(1)
        flattened_probs = torch.nn.Softmax(dim=1)(flatten_logits)
        probs = flattened_probs.reshape(logits.shape)

        return logits.cpu().numpy(), probs.cpu().numpy()
    
    def visualize_positional_weighting(self):
        plt.imshow(self.positional_weighting.squeeze().cpu().detach().numpy())
        plt.gca().invert_yaxis()
        plt.show()