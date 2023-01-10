import torch
from torch import nn
import numpy as np

class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int, frame_stack: int):
        super(SnakeBrain, self).__init__()

        #Part 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3*frame_stack, 16, 3, padding = (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        """nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
         )"""

        conv_output = 1152#4624

        #Value of state
        self.V = nn.Sequential(
            nn.Linear(conv_output, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),   
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )   
        #Impact of action
        self.A = nn.Sequential(
            nn.Linear(conv_output, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),   
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output)
        )   
        
        
    def forward(self, x: torch.Tensor, return_separate = False):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = torch.flatten(self.conv1(x), 1)
        A = self.A(x)
        V = self.V(x)
        if return_separate:
            return V + (A-A.mean(dim = 1, keepdim = True)), V, A
        return V + (A-A.mean(dim = 1, keepdim = True))
