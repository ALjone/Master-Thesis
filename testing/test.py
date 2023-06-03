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
        blocks.append(conv(observation_space.shape[1]+1, 32, kernel_size=5, padding = 2))
        blocks.append(nn.LeakyReLU())
        for _ in range(10-2):
            blocks.append(layer(32, 32, kernel_size=5))
            if use_batch_norm:
                blocks.append(bn(32))


        self.conv = nn.Sequential(*blocks)


        self.unit_output = conv(32, 1, 1)
        
        self.critic_output = nn.Sequential(nn.Linear(32, 64),
                                           nn.LeakyReLU(),
                                           nn.Linear(64, 32),
                                           nn.LeakyReLU(),
                                           nn.Linear(32, 1))
        
        self.positional_weighting = nn.Parameter(torch.ones(observation_space.shape[2:]).unsqueeze(0))
        self.temperature = nn.Parameter(torch.ones(1)) #NOTE: Actually just a scaling factor on the output logits

        print("Running with", self.count_parameters(), "parameters")


    def forward(self, observations: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observations = observations.squeeze()
        for i in range(len(observations.shape)-1):
            time = time.unsqueeze(1)

        global_features = torch.ones((observations.shape[0], 1) + tuple(observations.shape[-1] for _ in range(len(observations.shape)-2)), device=torch.device("cuda"))*time #Make it fit the observation, with 1 channel, to stack
        if observations.isnan().any():
            print("Found NaN in observation!!!")
            print(observations.isnan().sum().item(), "NaNs founds")

        x = torch.cat((observations, global_features), dim = 1)
        x = self.conv(x)

        action = (self.unit_output(x)*self.positional_weighting)/self.temperature


        critic = self.critic_output(nn.AvgPool2d(x.shape[-2])(x).flatten(1))

        return action.squeeze(), critic

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def visualize_dist(self, img, t):
        plt.cla()
        plt.close()
        with torch.no_grad():
            prob, _ = self(img, t)
        plt.imshow(prob[0].cpu().numpy())
        plt.show()


if __name__ == "__main__":
    b = torch.ones((2, 5, 30, 30)).to(torch.device("cuda"))
    c = torch.zeros((1, 5, 30, 30)).to(torch.device("cuda"))
    a = Agent(b, 2).to(torch.device("cuda"))
    a.load_state_dict(torch.load("models/model.t"))

    a.visualize_dist(b, torch.tensor([0, 0]).to(torch.device("cuda")))