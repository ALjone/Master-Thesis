from typing import Tuple
from gym import spaces
import matplotlib.pyplot as plt
from random_function import RandomFunction
from tilecoder import TileCoder
from GPY import GP
import torch
import numpy as np

from utils import rand

class BlackBox():
    def __init__(self, resolution = 30, domain = [-1, 1], batch_size = 128, num_init_points = 4, T = 120, kernels = None, dims = 3, use_GP = True, GP_learning_rate = 0.1, GP_training_iters = 200, approximate = False, expand_size = 50, noise = None, print_ = False):
        #TODO: Make it take in a device...
        if print_:
            print("Initialized with the following parameters:")
            print(f"  Resolution: {resolution} (number of grid cells in each dimension)")
            print(f"  Domain: {domain} (range of values for each dimension)")
            print(f"  Batch Size: {batch_size} (number of environments in each batch)")
            print(f"  Number of Initial Points: {num_init_points} (number of initial random points per environment)")
            print(f"  Total Time Steps: {T}")
            print(f"  Kernels: {kernels} (kernel functions for Gaussian Process)")
            print(f"  Dimensions: {dims} (number of dimensions in the function space)")
            print(f"  Use Gaussian Process: {use_GP}")
            print(f"  GP Learning Rate: {GP_learning_rate} (learning rate for Gaussian Process)")
            print(f"  GP Training Iterations: {GP_training_iters} (number of training iterations for Gaussian Process)")
            print(f"  Approximate: {approximate} (whether to use approximate Gaussian Process inference)")
            print(f"  Expand Size: {expand_size} (expansion size for storing actions and values for Gaussian Process)")
            print(f"  Noise: {noise} (noise level for Gaussian Process)")
            print()

        assert batch_size > 1, "Currently only batch size bigger than 1 supported. "
        #Set important variables
        self.num_init_points = num_init_points
        self.resolution = resolution
        self.batch_size = batch_size
        self.dims = dims
        self.use_GP = use_GP

        self.action_max = 1
        self.action_min = -1

        assert self.action_max == 1, "Fix transform action if you want to use action max other than 1"

        #Things for the env
        self.observation_space = spaces.Box(low=0, high=1, shape=
                    ((batch_size, 3) + tuple(resolution for _ in range(dims))), dtype=np.float32)
        self.action_space = spaces.Box(low = self.action_min, high = self.action_max, shape = (batch_size, dims), dtype=np.float32)

        self.reward_range = (0, 1) 
        
        #Set the total avaiable time
        self.T = T

        #Initialize the bounds
        self.x_min, self.x_max = domain[0], domain[1]

        self.steps = 0

        self.coder = TileCoder(resolution, domain, dims = dims) #NOTE: Sorta useless atm
        self.function_generator = RandomFunction((domain[0], domain[1]), resolution, batch_size, dims = dims)

        self.names = [str(i) for i in range(len(domain)//2)]


        self.time = torch.zeros(batch_size).to(torch.device("cuda"))
        self.grid = torch.zeros(self.observation_space.shape, dtype = torch.float32).to(torch.device("cuda"))
        self.params_for_time = {}
        for dim in range(dims):
            self.params_for_time[dim] = rand(1, 2, size=(self.batch_size, )).to(torch.device("cuda"))
        self.params_for_time["constant"] = rand(2, 4, size=(self.batch_size, )).to(torch.device("cuda"))

        self.max_time = self._t(torch.full_like(torch.zeros((batch_size, dims)), self.x_max).to(torch.device("cuda")), idx = torch.arange(start=0, end=batch_size)).to(torch.device("cuda"))
        assert self.max_time.shape[0] == batch_size

        self.actions_for_gp = [[]]*self.batch_size
        self.values_for_gp = [[]]*self.batch_size
        self.expand_size = expand_size #NOTE: This just says how much we expand the two above to
        self.batch_step = torch.zeros((batch_size)).to(torch.long).to(torch.device("cuda")) #The tracker for what step each "env" is at 
        self.best_prediction = torch.zeros((batch_size)).to(torch.device("cuda")) #Assumes all predictions are positive
        self.previous_closeness_to_max = torch.zeros((batch_size)).to(torch.device("cuda"))

        if self.use_GP:
            self.GP = GP(kernels, batch_size, domain, self.resolution, dims=dims, learning_rate=GP_learning_rate, training_iters=GP_training_iters, approximate=approximate)
        self.idx = torch.arange(start = 0, end = self.batch_size).to(torch.device("cuda"))

        self.episodic_returns = torch.zeros((batch_size)).to(torch.device("cuda"))

        self.reset()

    def _get_pred_true_max(self):
        true_min = self.function_generator.min
        true_max = self.function_generator.max-true_min
        pred_max = self.best_prediction-true_min
        return pred_max, true_max

    def _get_closeness_to_max(self):
        pred, true = self._get_pred_true_max()
        return pred/true

    def _t(self, action, idx = None):
        if idx is None: idx = self.idx
        #TODO: Is this correct? Why so much transformation here
        x_range = self.x_max-self.x_min
        res = 0
       
        for dim in range(self.dims):
            act = (action[:, dim]-self.x_min)/x_range
            res += self.params_for_time[dim][idx]*act
        res += self.params_for_time["constant"][idx] + torch.normal(0, 0.1, size = (idx.shape[0], )).to(torch.device("cuda"))

        return res

    def _check_init_points(self, idx):
        if idx.shape[0] == 0:
            print("????", idx)
            return
        #raise NotImplementedError("Needs to circumvent step, to only check init points for idx")
        for i in range(self.num_init_points):
            action = torch.tensor(self.action_space.sample()).to(torch.device("cuda")) #rand(self.action_min, self.action_max, (idx.shape[0], self.dims)) #self.action_space.sample()[idx].to(torch.device("cuda"))
            if len(action.shape) == 1: action = action.unsqueeze(0)
            #Transform from -1 to 1 -> current domain
            act = self._transform_actions([action[idx, i] for i in range(self.dims)])

            #Find the indices of the different overlapping boxes
            ind = self._find_indices(act)

            #Find the time and value for this action
            time = self._t(ind, idx)

            action_value = self.func_grid[(idx, ) + tuple(ind[:, i] for i in range(ind.shape[-1]))].squeeze()

            for num, i in enumerate(idx):
                self.actions_for_gp[i].append(act[num])
                self.values_for_gp[i].append(action_value[num] if len(idx) > 1 else action_value)

            self.grid[(idx, 2) + tuple(ind[:, i] for i in range(ind.shape[-1]))] = torch.maximum(time, self.grid[(idx, 2) + tuple(ind[:, i] for i in range(ind.shape[-1]))])

            #Update timestuff
            self.best_prediction[idx] = torch.maximum(self.best_prediction[idx], action_value)
            assert len(self.best_prediction.shape) == 1, f"Expected len 1, found: {self.best_prediction.shape}"
            self.previous_closeness_to_max[idx] = self._get_closeness_to_max()[idx]

            self.batch_step[idx] = self.batch_step[idx] + 1

    def _get_reward(self):
        #TODO: Getting from 94% to 96% should reward more than 30% to 40%
        r = self._get_closeness_to_max() - self.previous_closeness_to_max
        return r

    def reset(self, idx = None) -> torch.Tensor:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        if idx is None: idx = self.idx
        if len(idx.shape) == 0: idx = idx.unsqueeze(0)
        self.time[idx] = 0

        #Constants for the time function
        #NOTE: Max is 32827 seconds, min is 406
        for dim in range(self.dims):
            param = self.params_for_time[dim]
            param[idx] = rand(1, 2, size=idx.shape[0]).to(torch.device("cuda"))
            self.params_for_time[dim] = param
        constant = self.params_for_time["constant"]
        constant[idx] = rand(2, 4, size=idx.shape[0]).to(torch.device("cuda"))
        self.params_for_time["constant"]

        self.max_time[idx] = self._t(torch.full_like(torch.zeros((len(idx), self.dims)), self.x_max).to(torch.device("cuda")), idx)

        self.function_generator.reset(idx)
        self.func_grid = self.function_generator.matrix
        
        #self.actions_done = []
        self.previous_closeness_to_max[idx] = 0

        self.grid[idx] = 0
        self.best_prediction[idx] = 0

        self.batch_step[idx] = 0

        for i in idx:
            self.actions_for_gp[i] = []
            self.values_for_gp[i] = []

        self._check_init_points(idx)
        self._update_grid_with_GP(idx)

        self.episodic_returns[idx] = 0
        
        return self._get_state()


    def pad_sublists(self, list_of_lists):
        padded_lists = []
        for sublist in list_of_lists:
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

        mean, interval = self.GP.get_mean_std(self.pad_sublists(self.actions_for_gp), self.pad_sublists(self.values_for_gp), idx)

        self.grid[idx, 0] = mean
        self.grid[idx, 1] = interval

    def _get_state(self):
        #TODO: This uses illegal information in that it is providing the actual max of the time function
        new_grid = torch.clone(self.grid)
        #new_grid[:, 0] = new_grid[:, 0]/self.best_prediction[:, None, None, None]
        #new_grid[:, 1] = new_grid[:, 1]/self.best_prediction[:, None, None, None] #TODO: Is it correct to divide the std with the biggest prediction? Or should it be by biggest std? Ask Nello?
        new_grid[:, 2] = new_grid[:, 2]/self.T

        return new_grid, self.time

    def _find_indices(self, action):
        #print("Action:", action)
        #print("Idx:", self.coder[tuple(a for a in action)])
        return self.coder[action]

    def _transform_action(self, val, new_max, new_min):
        OldRange = (self.action_max - self.action_min)  
        NewRange = (new_max - new_min)
        return (((val - self.action_min) * NewRange) / OldRange) + new_min

    def _transform_actions(self, action):
        output = []
        for a in action:
            output.append(self._transform_action(a, self.x_max, self.x_min))
        return torch.stack(output, dim = 1).to(torch.device("cuda"))
        return self._transform_action(x, self.x_max, self.x_min), self._transform_action(y, self.x_max, self.x_min), self._transform_action(z, self.x_max, self.x_min)

    def step(self, action, transform = False) -> Tuple[torch.Tensor, float, bool]:
        """Completes the given action and returns the new map"""

        #Clip actions
        if transform:
            torch.tanh(action, out = action)

        #Transform from -1 to 1 -> current domain
        act = self._transform_actions([action[:, i] for i in range(self.dims)])

        #Find the indices of the different overlapping boxes
        ind = self._find_indices(act)


        #Find the time and value for this action
        time = self._t(ind)

        action_value = self.func_grid[(torch.arange(self.batch_size), ) + tuple(ind[:, i] for i in range(ind.shape[-1]))]


        assert action_value.shape[0] == self.batch_size

        #TODO: Look for bugs like the one that was here (with need arange)
        #Gather?
        for i in range(self.batch_size):
            self.actions_for_gp[i].append(act[i])
            self.values_for_gp[i].append(action_value[i])

        #Update all the different squares that the action affected
        self._update_grid_with_GP()

        self.grid[(slice(None), 2) + tuple(ind[:, i] for i in range(ind.shape[-1]))] = torch.maximum(time, self.grid[(slice(None), 2) + tuple(ind[:, i] for i in range(ind.shape[-1]))])

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
        #reward[~dones] = 0

        self.episodic_returns = self.episodic_returns + reward
        info = {"peak": pred_max/true_max, "episodic_returns": self.episodic_returns.clone(), "episodic_length" : self.batch_step.clone().to(torch.float)}

        if torch.sum(dones) > 0:
            self.reset(torch.nonzero(dones).squeeze())

        return self._get_state(), reward, dones, info

    def _display_axis(self, idx, axs, fig, data, title, invert = True):
        im = axs[idx].imshow(data)
        fig.colorbar(im, ax=axs[idx])
        if invert:
            axs[idx].invert_yaxis()
        axs[idx].set_title(title)

    def render(self, mode='human', close=False, batch_idx = 0):
        assert self.dims == 2, "Only supported for 2D"
        if mode == "human":
            if close:
                plt.cla()
                plt.close()
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))
            state, _ = self._get_state()
            self._display_axis(0, axs, fig, self.func_grid[batch_idx].reshape(self.resolution, self.resolution).cpu(), "Function")
            #TODO: Add
            #self._display_axis(1, axs, fig, self._t(self.vals[:, 0], self.vals[:, 1]).reshape(self.resolution, self.resolution), "Time")
            self._display_axis(1, axs, fig, state[batch_idx, 1].cpu(), "std for PPO", invert = True)
            self._display_axis(2, axs, fig, state[batch_idx, 0].cpu(), "Mean for PPO", invert = True)

            max_coords = torch.argmax(self.func_grid[batch_idx]).item()
            y_max, x_max = divmod(max_coords, self.resolution)

            for elem in self.actions_for_gp[batch_idx][self.num_init_points:]:
                a = self._find_indices(elem.unsqueeze(0)).squeeze()
                y, x = a[0], a[1]
                axs[0].scatter(x.cpu(), y.cpu(), c = "blue", linewidths=7)

            for elem in self.actions_for_gp[batch_idx][:self.num_init_points]:
                a = self._find_indices(elem.unsqueeze(0)).squeeze()
                y, x = a[0], a[1]
                axs[0].scatter(x.cpu(), y.cpu(), c = "red", linewidths=7)



            #TODO: Fix this..
            #y, x = self.actions_for_gp[batch_idx, -1].cpu()
            #if len(self.actions_for_gp[batch_idx]) > 2:
                #print("Last action idx pos", round(x, 2), round(y, 2))
                #axs[0].scatter(x, y, c = "green", linewidths=7)
                #axs[1].scatter(x, y, c = "green", linewidths=7)

            axs[0].scatter(x_max, y_max, c = "black", linewidths = 5)
            self._get_closeness_to_max()
            fig.suptitle(f"Percent of max at best guess: {round(self._get_closeness_to_max()[0].item(), 3)}\nPercent of max at last guess: {round(self.previous_closeness_to_max[0].item(), 3)}")

            plt.axis("off")
            plt.show()

        else:
            raise ValueError(f"Only mode 'human' supported, found: {mode}")