from typing import Tuple
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib as mpl
import io

class BlackBox(gym.Env):
    def __init__(self, resolution = 40, domain = [2, 12, 2, 12], u_v_range = 4, num_init_points = 2, T = 60):
        #Set important variables
        self.range = u_v_range
        self.num_init_points = num_init_points
        self.resolution = resolution
        

        #Things for the env
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (resolution, resolution, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype=np.float16)
        self.reward_range = (0, 1) 
        
        #Set the total avaiable time.
        self.T = T

        #Initialize the bounds
        self.x_min, self.x_max, self.y_min, self.y_max = domain[0], domain[1], domain[2], domain[3]

        self.steps = 0

        #Get a n by 2 array of all the possible points
        #TODO is this correct?
        x_bins, x_bin_size = np.linspace(self.x_min, self.x_max, self.resolution, retstep = True)
        y_bins, y_bin_size = np.linspace(self.y_min, self.y_max, self.resolution, retstep = True)

        self.vals =  np.array(np.meshgrid(x_bins, y_bins)).T.reshape(-1, 2)

        #Get bin size
        x_bin_size = x_bin_size/2
        y_bin_size = y_bin_size/2
        print(x_bin_size)

        if self.resolution%2 != 0:
            print("Resolution not divisible by 2, not gonna work so smoothly!!")

        #Create the bins used for the overlapping boxes when turning continous point into discrete label
        self.x_bins_left = np.linspace(self.x_min, self.x_max-x_bin_size, self.resolution//2, dtype=np.float64)
        self.x_bins_right = np.linspace(self.x_min+x_bin_size, self.x_max, self.resolution//2, dtype=np.float64)
        self.y_bins_left = np.linspace(self.y_min, self.y_max-y_bin_size, self.resolution//2)
        self.y_bins_right = np.linspace(self.y_min+y_bin_size, self.y_max, self.resolution//2)


        #Used for the logger
        self.all_actions = np.zeros((20000, 2))
        self.action_idx = 0

        self.reset()

    def _get_pred_true_max(self):
        true_min = self.min_
        true_max = self.max_-true_min
        pred_max = self.best_prediction-true_min
        return pred_max, true_max

    def _get_closeness_to_max(self):
        pred, true = self._get_pred_true_max()
        return pred/true

    def _max_of_current_function(self):
        return np.max(self.func_grid)

    def _min_of_current_function(self):
        return np.min(self.func_grid)

    def _f(self, x, y):
        #return self.x_sign*x + self.y_sign*y
        res = 0
        for u in range(-self.range, self.range+1):
            for v in range(-self.range, self.range+1):
                res += self._exp(u, v, x, y) if u != 0 and v != 0 else self.alpha[u, v]

        return np.abs(res)

    def _exp(self, u, v, x, y):
        P = 30
        return self.alpha[u, v]*np.exp((2*np.pi*1j*(u*x+v*y))/P)

    def _t(self, x, y):
        x_range = self.x_max-self.x_min
        y_range = self.y_max-self.y_min
        x = (x-self.x_min)/x_range
        y = (y-self.y_min)/y_range

        return self.a * x + self.b * y + self.c + np.random.normal(0, 0.3)

    def _check_init_points(self):
        for _ in range(self.num_init_points):
            self.step(self.action_space.sample(), True)
        self.steps = 0

    def _get_reward(self, force = False):
        #TODO: Tweak this...
        r = self._get_closeness_to_max() - self.previous_closeness_to_max
        return max(r, 0)
        #if r <= 0:
        #    return 0
        #return np.exp(r).item()-1


        if self.time > self.T or force:
            pred_max, true_max = self._get_pred_true_max()
            return np.exp((pred_max/true_max).item())
        return 0

    def reset(self) -> np.ndarray:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        self.time = 0

        #Constants for the fourier series
        self.alpha = np.random.uniform(-20, 20, (self.range*2+1, self.range*2+1))

        #Constants for the time function
        #NOTE: Max is 32827 seconds, min is 406
        self.a = np.random.uniform(1, 2.5)
        self.b = np.random.uniform(1, 2.5)
        self.c = np.random.uniform(2, 4)

        self.func_grid = self._f(self.vals[:, 0], self.vals[:, 1])
        self.max_ = self._max_of_current_function()
        self.min_ = self._min_of_current_function()
        
        self.actions_done = []
        self.previous_closeness_to_max = 0

        self.grid = np.zeros((3, self.resolution, self.resolution), dtype = np.uint8)

        self.best_prediction = 0

        self._check_init_points()

        return self.get_state()

    def get_state(self):
        #TODO: This uses illegal information in that it is providing the actual max of the time function
        new_grid = self.grid.copy()
        new_grid[0] = ((new_grid[0] - 0) * (1/(self.best_prediction - 0) * 255))
        new_grid[2] = ((new_grid[2] - 0) * (1/(9 - 0) * 255))

        return new_grid.transpose(1, 2, 0)

    def valid_moves(self):
        return 1 - self.grid[1]

    def _find_indices(self, x, y):
        indices = []

        left_x_ind = np.digitize(x, self.x_bins_left)*2
        right_x_ind = np.digitize(x, self.x_bins_right)*2+1

        left_y_ind = np.digitize(y, self.y_bins_left)*2
        right_y_ind = np.digitize(y, self.y_bins_right)*2+1

        for x_ind in [left_x_ind, right_x_ind]:
            for y_ind in [left_y_ind, right_y_ind]:
                if x_ind > 0 and y_ind > 0 and x_ind <= self.resolution and y_ind <= self.resolution:
                    indices.append((x_ind-1, y_ind-1))

        print("Number of squares:", len(indices))
        print("Indicies:", indices)
        print("x, y", x, y)
        print()
        return indices

    def _find_indices(self, x, y):
        indices = np.column_stack((np.digitize(x, self.x_bins_left), np.digitize(y, self.y_bins_left)))
        print("Indicies:", indices)
        indices = indices[(indices[:, 0] > 0) & (indices[:, 1] > 0) & (indices[:, 0] <= self.resolution) & (indices[:, 1] <= self.resolution), :]
        print("Number of squares:", len(indices))
        print("x, y", x, y)
        print()
        return indices - 1

    def _find_indices(self, x, y):
        #TODO optimize this, only need to call digitize 4 times, not 8
        indices = []
        for x_bin in [self.x_bins_left, self.x_bins_right]:
            for y_bin in [self.y_bins_left, self.y_bins_right]:
                x_ind, y_ind = np.digitize(x, x_bin), np.digitize(y, y_bin)
                if x_ind > 0 and y_ind > 0 and x_ind <= self.resolution and y_ind <= self.resolution:
                    indices.append((x_ind-1, y_ind-1))

        print("Number of squares:", len(indices))
        print("Indicies:", indices)
        print("x, y", x, y)
        print()

        return indices

    def _transform_action(self, val, new_max, new_min):
        OldRange = (1 - (-1))  
        NewRange = (new_max - new_min)
        return (((val - (-1)) * NewRange) / OldRange) + new_min

    def _transform_actions(self, x, y):
        return self._transform_action(x, self.x_max, self.x_min), self._transform_action(y, self.y_max, self.y_min)

    def step(self, action, init=False) -> Tuple[np.ndarray, float, bool]:
        """Completes the given action and returns the new map"""
        if not -1 <= action[0] <= 1 or not -1 <= action[1] <= 1:
            raise ValueError(f"Received invalid action={action} which is not part of the action space")

        #Transform from -1 to 1 -> current domain
        x, y = self._transform_actions(*action)

        #Find the indices of the different overlapping boxes
        indicies = self._find_indices(x, y)

        #Log the time for this action
        time = self._t(x, y)

        action_value = self._f(x, y)

        #Update all the different squares that the action affected
        for x_ind, y_ind in indicies:
            self.grid[0, x_ind, y_ind] = max(action_value, self.grid[0, x_ind, y_ind])
            
            self.grid[1, x_ind, y_ind] = 1
            self.grid[2, x_ind, y_ind] = max(time, self.grid[2, x_ind, y_ind])

        #Update timestuff
        self.steps += 1
        self.time += time
        if action_value > self.best_prediction:
            self.best_prediction = action_value
        #Get the reward and set some values used to calculate reward
        reward = self._get_reward()
        self.previous_closeness_to_max = self._get_closeness_to_max()

        #Log the actions done
        self.actions_done.append(((x-self.x_min)*(self.resolution-1)/(self.x_max-self.x_min), (y-self.y_min)*(self.resolution-1)/(self.y_max-self.y_min)))

        if not init:
            self.all_actions[self.action_idx%self.all_actions.shape[0]] = (x, y)
            self.action_idx += 1

        pred_max, true_max = self._get_pred_true_max()
        return self.get_state(), reward, bool(self.time > self.T), {"pred_max": pred_max, "true_max": true_max}


    def _display_axis(self, idx, axs, fig, data, title, invert = True):
        im = axs[idx].imshow(data)
        fig.colorbar(im, ax=axs[idx])
        if invert:
            axs[idx].invert_yaxis()
        axs[idx].set_title(title)

    def render(self, mode='human', close=False):
        if mode == "human":
            if close:
                plt.cla()
                plt.close()
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))
            self._display_axis(0, axs, fig, self.func_grid.reshape(self.resolution, self.resolution), "Function")
            self._display_axis(1, axs, fig, self._t(self.vals[:, 0], self.vals[:, 1]).reshape(self.resolution, self.resolution), "Time")
            self._display_axis(2, axs, fig, self.get_state()[:, :, 0], "Features for PPO", invert = True)

            max_coords = np.argmax(self.func_grid)
            y_max, x_max = divmod(max_coords, self.resolution)

            for y, x in self.actions_done[:self.num_init_points]:
                axs[0].scatter(x, y, c = "red", linewidths=7)
                axs[1].scatter(x, y, c = "red", linewidths=7)

            for y, x in self.actions_done[self.num_init_points:-1]:
                axs[0].scatter(x, y, c = "blue", linewidths=7)
                axs[1].scatter(x, y, c = "blue", linewidths=7)

            y, x = self.actions_done[-1]
            if len(self.actions_done) > 2:
                print("Last action idx pos", round(x, 2), round(y, 2))
                axs[0].scatter(x, y, c = "green", linewidths=7)
                axs[1].scatter(x, y, c = "green", linewidths=7)

            axs[0].scatter(x_max, y_max, c = "black", linewidths = 5)
            axs[1].scatter(x_max, y_max, c = "black", linewidths = 5)

            plt.axis("off")
            plt.show()
        #This else is sorta hacky, this is if you want the action dist
        else:
            
            #Create a figure for the action distribution, and then return the array for it
            fig, ax = plt.subplots()
            if self.action_idx > self.all_actions.shape[0]:
                actions = self.all_actions
            else:
                actions = self.all_actions[:self.action_idx]
            img = ax.hist2d(actions[:, 0], actions[:, 1], bins = [np.arange(self.x_min, self.x_max, (self.x_max-self.x_min)/self.resolution),
            np.arange(self.y_min, self.y_max, (self.y_max-self.y_min)/self.resolution)], norm=mpl.colors.LogNorm())
            fig.colorbar(img[3], ax=ax)

            #Shamelessly stole this from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)

            w, h = fig.canvas.get_width_height()
            plt.close()
            return data.reshape((int(h), int(w), -1))