import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.linear(self.cnn(observations))
