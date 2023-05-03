import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Normal, Beta
import numpy as np


class Agent(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, dims = 3):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.actor_cnn = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.actor_cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]+1

        self.action_mean = nn.Linear(n_flatten, dims)
        self.action_logstd = nn.Parameter(torch.ones(dims, ))#nn.Linear(n_flatten, 3)

        self.critic_cnn = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic = nn.Linear(n_flatten, 1)

        print("Running with", self.count_parameters(), "parameters")


    def forward(self, observations: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observations = observations.squeeze()
        x = self.actor_cnn(observations)
        x = torch.concatenate((x, time.unsqueeze(1)), dim = 1)
        action_mean = self.action_mean(x) #Batch dim, (Mean, Std), (x, y, z)
        action_std = torch.exp(self.action_logstd(x))

        if torch.any(torch.isnan(x)):
            print("Found", torch.isnan(x).sum(), "NaNs in action mean")
            print("Max in observation:", torch.max(observations))
        if torch.any(torch.isnan(action_mean)):
            print("Found", torch.isnan(action_mean).sum(), "NaNs in action mean")
        if torch.any(torch.isnan(action_std)):
            print("Max found in x:", torch.max(x))
            print("Found", torch.isnan(action_std).sum(), "NaNs in action std")

        x = self.critic_cnn(x)
        x = torch.concatenate((x, time.unsqueeze(1)), dim = 1)
        critic_output = self.critic(x)

        return action_mean, action_std, critic_output


    def get_value(self, x, time):
        _, _, value = self(x, time)
        return value

    def get_action_and_value(self, img, time, action=None):
        
        action_mean, action_std, critic_output = self(img, time)

        std = action_std.exp()

        normal = Normal(action_mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        
        if action is None:
            action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(1)

        return action, log_prob, normal.entropy().sum(1), critic_output
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    