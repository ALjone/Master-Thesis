import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


class Agent(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, dims = 3):
        super().__init__()
        self.dims = dims
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

        self.action_mean = nn.Sequential(nn.Linear(n_flatten, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, dims))
        self.action_logstd = nn.Sequential(nn.Linear(n_flatten, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, dims))#nn.Parameter(torch.ones((dims, ))*0.2)#nn.Linear(n_flatten, 3)

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
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            conv(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.critic = nn.Sequential(nn.Linear(n_flatten, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 1))

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

    def get_action_and_value(self, img, time, x_t=None):
        #TODO!!! We take in action, but find logprob of x_t?
        action_mean, action_std, critic_output = self(img, time)

        normal = Normal(action_mean, action_std)
        if x_t is None:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(1)

        return x_t, log_prob, normal.entropy().sum(1), critic_output, action_std
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def visualize_dist(self, img, t):
        action_mean, action_std, _ = self(img, t)
        for mu, variance in zip(action_mean[0].squeeze().cpu().numpy(), action_std[0].squeeze().cpu().numpy()):
            plt.cla()
            plt.close()
            sigma = variance
            print("Variance:", sigma)
            print("Mu:", mu)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, np.tanh(stats.norm.pdf(x, mu, sigma)))
            plt.show()