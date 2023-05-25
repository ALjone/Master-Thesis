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

    def __init__(self, observation_space: spaces.Box, dims):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        if dims == 3:
            conv = nn.Conv3d
        elif dims == 2:
            conv = nn.Conv2d
        else: 
            raise ValueError("Only dims = 2 or dims = 3 is currently supported")
        
        self.actor_cnn = nn.Sequential(
            conv(3, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.actor_cnn(
                torch.as_tensor(observation_space.sample()[0, None]).float()
            ).shape[1]+1

        self.action_mean = nn.Sequential((nn.Linear(n_flatten, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, dims)))
        
        self.action_logstd = nn.Sequential((nn.Linear(n_flatten, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, dims)))

        self.critic_cnn = nn.Sequential(
            conv(3, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.critic = nn.Sequential((nn.Linear(n_flatten, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, dims)))

        print("Running with", self.count_parameters(), "parameters")


    def forward(self, observations: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observations = observations.squeeze()
        x = self.actor_cnn(observations)
        x = torch.concatenate((x, time.unsqueeze(1)), dim = 1)
        action_mean = self.action_mean(x) #Batch dim, (Mean, Std), (x, y, z)
        action_std = torch.exp(self.action_logstd(x))

        x = self.critic_cnn(observations)
        x = torch.concatenate((x, time.unsqueeze(1)), dim = 1)
        critic_output = self.critic(x)
        return action_mean, action_std, critic_output


    def get_value(self, x, time):
        _, _, value = self(x, time)
        return value

    def get_action_and_value(self, img, time, action=None):
        
        action_mean, action_std, critic_output = self(img, time)
        probs = Normal(action_mean, action_std)
        if torch.any(torch.isnan(img)):
            print("Found", torch.isnan(img).sum(), "NaNs in input")

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), critic_output, action_std
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    