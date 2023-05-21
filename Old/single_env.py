from typing import Tuple
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
from tilecoder import TileCoder
from sklearn_GP import GP
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Exponentiation, Matern, RationalQuadratic, ExpSineSquared
from scipy.stats import norm
def EI(u, std, biggest, e = 0.01):
    if std <= 0:
        print("std under 0")
        return 0
    Z = (u-biggest-e)/std
    return (u-biggest-e)*norm.cdf(Z)+std*norm.pdf(Z)

class BlackBox(gym.Env):
    def __init__(self, resolution = 20, domain = [2, 12, 2, 12, 2, 12], u_v_range = 2, num_init_points = 4, T = 60, kernels = None, acquisition = None):
        #Set important variables
        self.range = u_v_range
        self.num_init_points = num_init_points
        self.resolution = resolution
        

        #Things for the env
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (3, resolution, resolution, resolution), dtype=np.uint8)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3, ), dtype=np.float32)
        self.reward_range = (0, 1) 
        
        #Set the total avaiable time
        self.T = T

        #Initialize the bounds
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = domain[0], domain[1], domain[2], domain[3], domain[4], domain[5]

        self.steps = 0

        #Get a n by 2 array of all the possible points
        x_bins = np.linspace(self.x_min, self.x_max, self.resolution)
        y_bins = np.linspace(self.y_min, self.y_max, self.resolution)
        z_bins = np.linspace(self.z_min, self.z_max, self.resolution)

        self.vals =  np.array(np.meshgrid(x_bins, y_bins, z_bins)).T.reshape(-1, 3)
        if self.resolution%2 != 0:
            print("Resolution not divisible by 2, not gonna work so smoothly!!")

        self.coder = TileCoder(resolution, domain)
        #Used for the logger
        self.all_actions = np.zeros((20000, 2))
        self.action_idx = 0

        #For GP
        self.kernels = kernels if kernels is not None else [1*Matern(), 1*RBF()]
        self.acquisition = acquisition if acquisition is not None else EI
        self.names = [str(i) for i in range(len(domain)//2)]

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

    def _f(self, x, y, z):
        res = 0
        for u in range(-self.range, self.range+1):
            for v in range(-self.range, self.range+1):
                for w in range(-self.range, self.range+1):
                    res += self._exp(u, v, w, x, y, z) if u != 0 and v != 0 and w != 0 else self.alpha[u, v, w]

        return np.abs(res)

    def _exp(self, u, v, w, x, y, z):
        P = 30
        return self.alpha[u, v, w]*np.exp((2*np.pi*1j*(u*x+v*y+w*z))/P)

    def _t(self, x, y, z):
        x_range = self.x_max-self.x_min
        y_range = self.y_max-self.y_min
        z_range = self.z_max-self.z_min
        x = (x-self.x_min)/x_range
        y = (y-self.y_min)/y_range
        z = (z-self.z_min)/z_range

        return self.a * x + self.b * y + self.c*z + self.d + np.random.normal(0, 0.1)

    def _check_init_points(self):
        for _ in range(self.num_init_points):
            self.step(self.action_space.sample(), True)
        self.steps = 0

    def _get_reward(self):
        #TODO: Tweak this...
        r = self._get_closeness_to_max() - self.previous_closeness_to_max
        return max(r, 0)

    def reset(self) -> np.ndarray:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        self.time = 0

        #Constants for the fourier series
        self.alpha = np.random.uniform(-20, 20, (self.range*2+1, self.range*2+1, self.range*2+1))

        #Constants for the time function
        #NOTE: Max is 32827 seconds, min is 406
        self.a = np.random.uniform(1, 2)
        self.b = np.random.uniform(1, 2)
        self.c = np.random.uniform(1, 2)
        self.d = np.random.uniform(2, 4)

        self.max_time = self._t(self.x_max, self.y_max, self.z_max)

        self.func_grid = self._f(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        self.max_ = self._max_of_current_function()
        self.min_ = self._min_of_current_function()
        
        self.actions_done = []

        self.actions_for_gp = []
        self.values_for_gp = []
        self.previous_closeness_to_max = 0

        self.grid = np.zeros((3, self.resolution, self.resolution, self.resolution), dtype = np.float32)

        self.best_prediction = 0


        self._check_init_points()

        self.GP = GP(self.kernels, self.acquisition, [(self.x_min, self.x_max), (self.y_min, self.y_max), (self.z_min, self.z_max)],
                      self.resolution, self.names, use_tqdm = False, checked_points = np.array(self.actions_for_gp), values_found = np.array(self.values_for_gp))
        return self.get_state()

    def update_GP(self, coords, value):
        self.GP.update_points(coords, value)
        self.GP.get_next_point()

        self.grid[0] = self.GP.mean
        self.grid[1] = self.GP.std

    def get_state(self):
        #TODO: This uses illegal information in that it is providing the actual max of the time function
        new_grid = self.grid.copy()
        new_grid[0] /= self.best_prediction
        new_grid[2] /= self.max_time

        return (new_grid*255).astype(np.uint8)

    def valid_moves(self):
        return 1 - self.grid[1]

    def _find_indices(self, x, y, z):
        return self.coder[x, y, z]

    def _transform_action(self, val, new_max, new_min):
        OldRange = (1 - (-1))  
        NewRange = (new_max - new_min)
        return (((val - (-1)) * NewRange) / OldRange) + new_min

    def _transform_actions(self, x, y, z):
        return self._transform_action(x, self.x_max, self.x_min), self._transform_action(y, self.y_max, self.y_min), self._transform_action(z, self.z_max, self.z_min)

    def step(self, action, init=False) -> Tuple[np.ndarray, float, bool]:
        #TODO: Use this init to not update GP, so we can do that after
        """Completes the given action and returns the new map"""
        if not -1 <= action[0] <= 1 or not -1 <= action[1] <= 1:
            raise ValueError(f"Received invalid action={action} which is not part of the action space")

        #Transform from -1 to 1 -> current domain
        x, y, z = self._transform_actions(*action)

        #Find the indices of the different overlapping boxes
        (x_ind, y_ind, z_ind) = self._find_indices(x, y, z)

        #Find the time and value for this action
        time = self._t(x, y, z)

        action_value = self._f(x, y, z)

        #Update all the different squares that the action affected
        if not init:
            self.update_GP(action, action_value)

        #TODO: Fix this. 0 and 1 layers are handled in GP. Fix means remove?
        #self.grid[0, x_ind, y_ind, z_ind] = max(action_value, self.grid[0, x_ind, y_ind, z_ind])
        
        #self.grid[1, x_ind, y_ind, z_ind] = 1
        self.grid[2, x_ind, y_ind, z_ind] = max(time, self.grid[2, x_ind, y_ind, z_ind])

        #Update timestuff
        self.steps += 1
        self.time += time
        self.best_prediction = max(self.best_prediction, action_value)
        #Get the reward and set some values used to calculate reward
        reward = self._get_reward()
        self.previous_closeness_to_max = self._get_closeness_to_max()

        #Log the actions done
        self.actions_done.append(((x-self.x_min)*(self.resolution-1)/(self.x_max-self.x_min), (y-self.y_min)*(self.resolution-1)/(self.z_max-self.z_min), (z-self.z_min)*(self.resolution-1)/(self.z_max-self.z_min)))

        self.actions_for_gp.append(action)
        self.values_for_gp.append(action_value)

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
                #print("Last action idx pos", round(x, 2), round(y, 2))
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
            #fig.colorbar(img[3], ax=ax)

            #Shamelessly stole this from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)

            w, h = fig.canvas.get_width_height()
            plt.close()
            return data.reshape((int(h), int(w), -1))