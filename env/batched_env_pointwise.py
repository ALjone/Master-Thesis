from typing import Tuple
from gym import spaces
import matplotlib.pyplot as plt
from functions.random_functions.random_functions import RandomFunction
from functions.time_functions.time_functions import TimeFunction
from env.tilecoder import TileCoder
from env.GPY import GP
import torch
import numpy as np
from utils import rand

class MultiDiscrete2DActionSpace(spaces.MultiDiscrete):
    def __init__(self, n, resolution, dims):
        super().__init__([resolution for _ in range(n * dims)])
        self.n = n
        self.dims = dims

    def sample(self):
        return np.array(super().sample()).reshape((self.n, self.dims))

    @property
    def shape(self):
        return (self.n, self.dims)

class BlackBox():
    def __init__(self, config):
        #TODO: Make it take in a device...
        if config.verbose > 0:
            print("Initialized with the following parameters:")
            print(f"  Resolution: {config.resolution} (number of grid cells in each dimension)")
            print(f"  Domain: {config.domain} (range of values for each dimension)")
            print(f"  Batch Size: {config.batch_size} (number of environments in each batch)")
            print(f"  Number of Initial Points: {config.num_init_points} (number of initial random points per environment)")
            print(f"  Total Time Step range: {config.T_range}")
            print(f"  Dimensions: {config.dims} (number of dimensions in the function space)")
            print(f"  Use Gaussian Process: {config.use_GP}")
            print(f"  GP Learning Rate: {config.GP_learning_rate} (learning rate for Gaussian Process)")
            print(f"  GP Training Iterations: {config.GP_training_iters} (number of training iterations for Gaussian Process)")
            print(f"  Approximate: {config.approximate} (whether to use approximate Gaussian Process inference)")
            print(f"  Expand Size: {config.expand_size} (expansion size for storing actions and values for Gaussian Process)")
            print(f"  Noise: {config.noise} (noise level for Gaussian Process)")
            print(f"  Use time in state: {config.use_time} (Whether the state includes information about the sampling time")
            print()

        assert config.batch_size > 1, "Currently only batch size bigger than 1 supported."
        #Set important variables
        self.num_init_points = config.num_init_points
        self.resolution = config.resolution
        self.batch_size = config.batch_size
        self.dims = config.dims
        self.use_GP = config.use_GP
        self.use_time_in_state = config.use_time
        self.log_reward = config.log_reward
        self.config = config
        self.time_dims = config.time_dims

        self.action_max = 1
        self.action_min = -1
        self.idx = torch.arange(start = 0, end = self.batch_size).to(torch.device("cpu"))

        assert self.action_min == -1 and self.action_max == 1, "Action range needs to be (-1, 1), otherwise fix step"

        assert self.action_max == 1, "Fix transform action if you want to use action max other than 1"

        #Things for the env
        #Mean, STD, timeportion in point, time spent (constant), best prediction globally (constant), max time
        self.observation_space = spaces.Box(low=0, high=1, shape=((config.batch_size, 6 if self.use_time_in_state else 4) + tuple(config.resolution for _ in range(config.dims))), dtype=np.float32)
        self.action_space = MultiDiscrete2DActionSpace(self.batch_size, self.resolution, self.dims)

        #Initialize the grid. Always keep track of time information, even if we don't return it in state
        shape = list(self.observation_space.shape)
        shape[1] += (0 if self.use_time_in_state else 1)
        self.grid = torch.zeros(shape , dtype = torch.float32).to(torch.device("cpu"))

        self.reward_range = (0, 1) 
        
        #Set the total avaiable time
        self.T_min = config.T_range[0]
        self.T_max = config.T_range[1]
        self.T = torch.zeros(self.batch_size).to(torch.device("cpu"))

        #Initialize the bounds
        self.x_min, self.x_max = config.domain[0], config.domain[1]

        self.steps = 0

        self.coder = TileCoder(config.resolution, config.domain, dims = config.dims) #NOTE: Sorta useless atm
        self.function_generator = RandomFunction(config)
        self.time_grid = self._time_grid()

        #How much time has been spent so far
        self.time = torch.zeros(config.batch_size).to(torch.device("cpu"))

        self.params_for_time = {}

        self.actions_for_gp = [[]]*self.batch_size
        self.values_for_gp = [[]]*self.batch_size
        self.checked_points = torch.zeros((config.batch_size, ) + tuple(self.resolution for _ in range(config.dims)))
        self.expand_size = config.expand_size #NOTE: This just says how much we expand the two above to

        self.batch_step = torch.zeros((config.batch_size)).to(torch.long).to(torch.device("cpu")) #The tracker for what step each "env" is at 
        self.best_prediction = torch.zeros((config.batch_size)).to(torch.device("cpu")) #Assumes all predictions are positive
        self.previous_closeness_to_max = torch.zeros((config.batch_size)).to(torch.device("cpu"))

        if self.use_GP:
            self.GP = GP(config)

        self.episodic_returns = torch.zeros((config.batch_size)).to(torch.device("cpu"))

        self.reset()

    def _get_pred_true_max(self):
        true_min = self.function_generator.min
        true_max = self.function_generator.max-true_min
        pred_max = self.best_prediction-true_min
        return pred_max, true_max

    def _get_closeness_to_max(self):
        pred, true = self._get_pred_true_max()
        return pred/true

    def _time_grid(self, idx = None):
        if idx is None: idx = self.idx

        num_dims = self.dims
        x = torch.linspace(0, self.x_max-self.x_min, self.resolution, device=torch.device("cpu"))  # Generate equally spaced values between -1 and 1
        grids = torch.meshgrid(*([x] * num_dims))  # Create grids for each dimension
        points = torch.stack(grids, dim=-1)  # Stack grids along the last axis to get points

        result = torch.zeros_like(points[..., 0], device=torch.device("cpu"))  # Initialize result tensor
        for i in range(self.time_dims):
            exponent = rand(self.config.polynomial_range[0], self.config.polynomial_range[1], 1)
            coefficient = rand(self.config.linear_range[0], self.config.linear_range[1], 1)
            result += coefficient*(points[..., i]**exponent)

        shape = [len(idx)] + [-1]*num_dims
        
        return (result+rand(self.config.constant_range[0], self.config.constant_range[1], 1)).expand(shape)

    def _check_init_points(self, idx):
        if idx.shape[0] == 0:
            print("????", idx)
            return
        for i in range(self.num_init_points):
            ind = torch.tensor(self.action_space.sample()).to(torch.device("cpu"))[idx] #rand(self.action_min, self.action_max, (idx.shape[0], self.dims)) #self.action_space.sample()[idx].to(torch.device("cpu"))
            if len(ind.shape) == 1: ind = ind.unsqueeze(0)
            #Transform from -1 to 1 -> current domain
            act = (ind-(self.resolution//2))/(self.resolution//2)

            action_value = self.func_grid[(idx, ) + tuple(ind[:, i] for i in range(ind.shape[-1]))].squeeze()

            for num, i in enumerate(idx):
                self.actions_for_gp[i].append(act[num])
                self.values_for_gp[i].append(action_value[num] if len(idx) > 1 else action_value)

            #Update timestuff
            self.best_prediction[idx] = torch.maximum(self.best_prediction[idx], action_value)
            assert len(self.best_prediction.shape) == 1, f"Expected len 1, found: {self.best_prediction.shape}"
            self.previous_closeness_to_max[idx] = self._get_closeness_to_max()[idx]

            self.batch_step[idx] = self.batch_step[idx] + 1

    def _get_reward(self):
        #TODO: Getting from 94% to 96% should reward more than 30% to 40%
        #r = self._get_closeness_to_max() - self.previous_closeness_to_max
        pred_max, true_max = self._get_pred_true_max()
        simple_regret = true_max - pred_max
        if self.log_reward:
            simple_regret = torch.clip(simple_regret, min = 1e-5)
            return -torch.log10(simple_regret)
        return -simple_regret

        return r

    def reset(self, idx = None) -> torch.Tensor:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        if idx is None: idx = self.idx
        if len(idx.shape) == 0: idx = idx.unsqueeze(0)
        self.time[idx] = 0
        self.T[idx] = rand(self.T_min, self.T_max, len(idx))

        self.time_grid[idx] = self._time_grid(idx)
        self.function_generator.reset(idx)
        self.func_grid = self.function_generator.matrix
        
        #self.actions_done = []
        self.previous_closeness_to_max[idx] = 0

        self.grid[idx] = 0
        self.best_prediction[idx] = 0

        self.batch_step[idx] = 0

        self.grid[idx, 2] = self.time_grid[idx]

        for i in idx:
            self.actions_for_gp[i] = []
            self.values_for_gp[i] = []

        self._check_init_points(idx)
        self._update_grid_with_GP(idx)

        self.episodic_returns[idx] = 0
        
        return self._get_state()


    def pad_sublists(self, list_of_lists, idx):
        #Thanks to ChatGPT
        padded_lists = []
        for i in idx:
            sublist = list_of_lists[i]
            sublist_len = len(sublist)
            num_copies = self.expand_size // sublist_len
            padding_len = self.expand_size % sublist_len
            padded_sublist = sublist * num_copies + sublist[:padding_len]
            padded_lists.append(torch.stack(padded_sublist))

        return torch.stack(padded_lists)

    def _update_grid_with_GP(self, idx = None):
        if not self.use_GP:
            return

        #Normalize all self.values_for_gp. But should be fixed by just choosing a reasonable distribution to sample from
        if idx is None: idx = self.idx

        mean, std, _, _ = self.GP.get_mean_std(self.pad_sublists(self.actions_for_gp, idx), self.pad_sublists(self.values_for_gp, idx), idx)
        self.grid[idx, 0] = mean
        self.grid[idx, 1] = std

    def _get_state(self):
        if self.use_time_in_state:
            #Mean, STD, timeportion in point, time spent (constant), best prediction globally (constant)
            new_grid = torch.clone(self.grid)
            if self.dims == 2:
                new_grid[:, 2] /= self.T.unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 3] = (self.time/self.T).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 4] = self.best_prediction.unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 5] = (self.time_grid[:, -1, -1]/self.T).unsqueeze(-1).unsqueeze(-1)
            elif self.dims == 3:
                new_grid[:, 2] /= self.T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 3] = (self.time/self.T).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 4] = self.best_prediction.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 5] = (self.time_grid[:, -1, -1, -1]/self.T).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise NotImplementedError("Fix unsqueeze to work in n dims")
            assert new_grid.shape[1] == 6
            return new_grid
        else:
            #Mean, STD, time spent (constant), best prediction globally (constant)
            mask = torch.ones(5).to(torch.bool)
            mask[2] = 0
            new_grid = torch.clone(self.grid[:, mask])
            if self.dims == 2:
                new_grid[:, 2] = (self.time/self.T).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 3] = self.best_prediction.unsqueeze(-1).unsqueeze(-1)
            elif self.dims == 3:
                new_grid[:, 2] = (self.time/self.T).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                new_grid[:, 3] = self.best_prediction.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise NotImplementedError("Fix unsqueeze to work in n dims")
            assert new_grid.shape[1] == 4
            return new_grid

    def _find_indices(self, action):
        return self.coder[action]

    def _transform_action(self, val, new_max, new_min):
        OldRange = (self.action_max - self.action_min)  
        NewRange = (new_max - new_min)
        return (((val - self.action_min) * NewRange) / OldRange) + new_min

    def _transform_actions(self, action):
        output = []
        for a in action:
            output.append(self._transform_action(a, self.x_max, self.x_min))
        return torch.stack(output, dim = 1).to(torch.device("cpu"))

    def step(self, action, isindex = True) -> Tuple[torch.Tensor, float, bool]:
        """Completes the given action and returns the new map"""
        if not isindex:
            raise NotImplementedError("Only index is implemented as of now, though you can use ._find_indicies to transform them")
            #Transform from -1 to 1 -> current domain
            act = self._transform_actions([action[:, i] for i in range(self.dims)])

            #Find the indices of the different overlapping boxes
            ind = self._find_indices(act)
        else:
            ind = action
            act = (action-(self.resolution//2))/(self.resolution//2)

        #Find the time and value for this action
        time = self.grid[(torch.arange(self.batch_size), 2) + tuple(ind[:, i] for i in range(ind.shape[-1]))]
        action_value = self.func_grid[(torch.arange(self.batch_size), ) + tuple(ind[:, i] for i in range(ind.shape[-1]))]
        assert action_value.shape[0] == self.batch_size

        #TODO: Look for bugs like the one that was here (with need arange)
        #Gather?
        for i in range(self.batch_size):
            self.actions_for_gp[i].append(act[i])
            self.values_for_gp[i].append(action_value[i])

        #Update all the different squares that the action affected
        self._update_grid_with_GP()

        #Update timestuff
        self.time = self.time + time
        self.best_prediction = torch.maximum(self.best_prediction, action_value)
        assert len(self.best_prediction.shape) == 1, f"Expected len 1, found: {self.best_prediction.shape}"
        #Get the reward and set some values used to calculate reward
        reward = self._get_reward()
        self.previous_closeness_to_max = self._get_closeness_to_max()

        pred_max, true_max = self._get_pred_true_max()

        self.batch_step = self.batch_step + 1
        dones = self.time > self.T
        reward[~dones] = 0

        self.episodic_returns = self.episodic_returns + reward
        info = {"peak": pred_max/true_max, "episodic_returns": self.episodic_returns.clone(), "episodic_length" : self.batch_step.clone().to(torch.float)-self.num_init_points,
                "function_classes": self.function_generator.function_classes}

        if torch.sum(dones) > 0:
            self.reset(torch.nonzero(dones).squeeze())

        return self._get_state(), reward, dones, info

    def _display_axis(self, idx, axs, fig, data, title, invert = True):
        im = axs[0, idx].imshow(data)
        fig.colorbar(im, ax=axs[0, idx])
        if invert:
            axs[0, idx].invert_yaxis()
        axs[0, idx].set_title(title)

    def render(self, mode='human', show = True, batch_idx = 0, additional = {}):
        assert self.dims == 2, "Only supported for 2D"
        if mode == "human":
            fig, axs = plt.subplots(2, 7 - (0 if self.use_time_in_state else 2), figsize=(20, 10))
            state = self._get_state()
            self._display_axis(0, axs, fig, self.func_grid[batch_idx].reshape(self.resolution, self.resolution).cpu(), "Function")
            #TODO: Add
            self._display_axis(1, axs, fig, state[batch_idx, 0].cpu(), "Mean for PPO", invert = True)
            self._display_axis(2, axs, fig, state[batch_idx, 1].cpu(), "std for PPO", invert = True)
            if self.use_time_in_state:
                self._display_axis(3, axs, fig, state[batch_idx, 2].cpu(), "Time", invert = True)
                self._display_axis(4, axs, fig, state[batch_idx, 3].cpu(), "Time spent", invert = True)
                self._display_axis(5, axs, fig, state[batch_idx, 4].cpu(), "Best found", invert = True)
                self._display_axis(6, axs, fig, state[batch_idx, 5].cpu(), "Max time", invert = True)
            else:
                self._display_axis(3, axs, fig, state[batch_idx, 2].cpu(), "Time spent", invert = True)
                self._display_axis(4, axs, fig, state[batch_idx, 3].cpu(), "Best found", invert = True)
                
            max_coords = torch.argmax(self.func_grid[batch_idx]).item()
            y_max, x_max = divmod(max_coords, self.resolution)

            for i, elem in enumerate(self.actions_for_gp[batch_idx][self.num_init_points:]):
                a = self._find_indices(elem.unsqueeze(0)).squeeze()
                y, x = a[0], a[1]
                axs[0, 0].scatter(x.cpu(), y.cpu(), c = "blue", linewidths=7, label = "Actions made" if i == 0 else None)

            for i, elem in enumerate(self.actions_for_gp[batch_idx][:self.num_init_points]):
                a = self._find_indices(elem.unsqueeze(0)).squeeze()
                y, x = a[0], a[1]
                axs[0, 0].scatter(x.cpu(), y.cpu(), c = "red", linewidths=7, label = "Initial points" if i == 0 else None)

            for i, (name, img) in enumerate(additional.items()):
                img = axs[1, i].imshow(img)
                axs[1, i].invert_yaxis() #TODO: Why?
                axs[1, i].set_title(name)
                fig.colorbar(img, ax=axs[1, i])


            axs[0, 0].scatter(x_max, y_max, c = "black", linewidths = 5)
            self._get_closeness_to_max()
            fig.suptitle(f"Percent of max at best guess: {round(self._get_closeness_to_max()[0].item(), 3)}\nPercent of max at last guess: {round(self.previous_closeness_to_max[0].item(), 3)}")

            plt.axis("off")
            fig.legend()
            plt.show()

        else:
            raise ValueError(f"Only mode 'human' supported, found: {mode}")