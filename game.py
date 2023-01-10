import torch
from typing import Tuple
import numpy as np

class BlackBox:
    def __init__(self, resolution, domain = [-5, 5, -5, 5], u_v_range = 2, num_init_points = 2, T = 50):
        """Initializes the game with the correct size and lifespan, and puts a head in the middle, as well as an apple randomly on the map"""
        self.range = u_v_range
        self.num_init_points = num_init_points
        self.resolution = resolution
        self.T = T

        self.x_min = domain[0]
        self.x_max = domain[1]
        self.y_min = domain[2]
        self.y_max = domain[3]
        self.points = {}

        self.steps = 0

        self.x_bins = np.linspace(self.x_min, self.x_max, self.resolution)
        self.y_bins = np.linspace(self.y_min, self.y_max, self.resolution)
        self.vals =  np.array(np.meshgrid(self.x_bins, self.y_bins)).T.reshape(-1, 2)

        self.reset()
        print("Average for T:", np.mean(self.t(self.vals[:, 0], self.vals[:, 1])))

    def get_reward(self, force = False):
        if self.time > self.T or force:
            true_min = self.min_
            true_max = self.max_-true_min
            pred_max = torch.max(self.grid)-true_min
            return np.exp((pred_max/true_max).item())
        return 0

    def max(self):
        return np.max(self.func_grid)

    def min(self):
        return np.min(self.func_grid)

    def f(self, x, y):
        res = 0
        for u in range(-self.range, self.range+1):
            for v in range(-self.range, self.range+1):
                res += self.exp(u, v, x, y) if u != 0 and v != 0 else self.alpha[u, v]

        return np.abs(res)


    def exp(self, u, v, x, y):
        return self.alpha[u, v]*np.exp(np.pi*1j*(u*x+v*y))

    def t(self, x, y):
        x_range = self.x_max-self.x_min
        y_range = self.y_max-self.y_min
        x = (x-self.x_min)/x_range
        y = (y-self.y_min)/y_range

        return self.a * x + self.b * y + self.c + np.random.normal(0, 0.3)

    def reset(self) -> np.ndarray:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        self.time = 0

        #Constants for the fourier series
        self.alpha = np.random.uniform(-1, 1, (self.range*2+1, self.range*2+1))

        #Constants for the time function
        self.a = np.random.uniform(0.5, 1.5)
        self.b = np.random.uniform(0.5, 1.5)
        self.c = np.random.uniform(2, 4)

        self.grid = torch.zeros((3, self.resolution, self.resolution))
        for _ in range(self.num_init_points):
            x = np.random.randint(0, self.resolution)
            y = np.random.randint(0, self.resolution)
            self.step((x, y))
        
        self.func_grid = self.f(self.vals[:, 0], self.vals[:, 1])
        self.max_ = self.max()
        self.min_ = self.min()
        
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return self.grid

    def valid_moves(self):
        return 1 - self.grid[1]

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        """Completes the given action and returns the new map"""
        x_ind, y_ind = action[0], action[1]
        #print(x_ind, y_ind)
        x = self.x_bins[x_ind]
        y = self.y_bins[y_ind]
        #x_ind, y_ind = np.digitize(x, self.x_bins), np.digitize(y, self.y_bins)
        #print(x, y)
        #print(x_ind, y_ind)
        self.grid[0, x_ind, y_ind] = self.f(x, y)    
        reward = self.get_reward() - 1 if self.grid[1, x_ind, y_ind] == 1 else self.get_reward()
        self.grid[1, x_ind, y_ind] = 1
        self.steps += 1
        self.grid[2, x_ind, y_ind] = self.t(x, y)
        self.time += self.grid[2, x_ind, y_ind]
        
        return self.get_state(), self.get_reward(), self.time > self.T
