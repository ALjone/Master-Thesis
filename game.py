from typing import Tuple
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib as mpl
import io

class BlackBox(gym.Env):
    def __init__(self, resolution = 41, domain = [2, 12, 2, 12], u_v_range = 4, num_init_points = 2, T = 60):
        """Initializes the game with the correct size and lifespan, and puts a head in the middle, as well as an apple randomly on the map"""
        #super.__init__()
        self.range = u_v_range
        self.num_init_points = num_init_points
        self.resolution = resolution
        #TODO fix bounds
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (resolution, resolution, 3), dtype=np.uint8)

        self.action_space = spaces.Box(low=np.array([domain[0], domain[2]]), high=np.array([domain[1], domain[3]]), dtype=np.float16)

        self.reward_range = (0, 1) 
        self.T = T

        self.x_min = domain[0]
        self.x_max = domain[1]
        self.y_min = domain[2]
        self.y_max = domain[3]
        self.points = {}

        self.steps = 0

        #Make bins
        x_bin_size = (self.x_max-self.x_min)/(self.resolution*2)
        y_bin_size = (self.y_max-self.y_min)/(self.resolution*2)

        self.x_bins_left = np.linspace(self.x_min, self.x_max-x_bin_size, self.resolution, dtype=np.float64)
        self.x_bins_right = np.linspace(self.x_min+x_bin_size, self.x_max, self.resolution, dtype=np.float64)
        self.y_bins_left = np.linspace(self.y_min, self.y_max-y_bin_size, self.resolution)
        self.y_bins_right = np.linspace(self.y_min+y_bin_size, self.y_max, self.resolution)

        x_bins = np.linspace(self.x_min, self.x_max, self.resolution)
        y_bins = np.linspace(self.y_min, self.y_max, self.resolution)

        self.vals =  np.array(np.meshgrid(x_bins, y_bins)).T.reshape(-1, 2)

        #Used for the logger
        self.all_actions = []

        self.reset()

    def get_pred_true_max(self):
        true_min = self.min_
        true_max = self.max_-true_min
        pred_max = np.max(self.grid[0])-true_min
        return pred_max, true_max

    def get_closeness_to_max(self):
        pred, true = self.get_pred_true_max()
        return pred/true

    def get_reward(self, force = False):
        #TODO: Tweak this...
        r = self.get_closeness_to_max() - self.previous_closeness_to_max
        if r <= 0:
            return 0
        return np.exp(r).item()-1


        if self.time > self.T or force:
            pred_max, true_max = self.get_pred_true_max()
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
        P = 30
        return self.alpha[u, v]*np.exp((2*np.pi*1j*(u*x+v*y))/P)

    def t(self, x, y):
        x_range = self.x_max-self.x_min
        y_range = self.y_max-self.y_min
        x = (x-self.x_min)/x_range
        y = (y-self.y_min)/y_range

        return self.a * x + self.b * y + self.c + np.random.normal(0, 0.3)

    def check_init_points(self):
        for _ in range(self.num_init_points):
            x = np.random.randint(self.x_min, self.x_max)
            y = np.random.randint(self.y_min, self.y_max)
            self.step((x, y), True)

        self.steps = 0

    def reset(self) -> np.ndarray:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        #Time spent this run
        self.time = 0

        #Constants for the fourier series
        self.alpha = np.random.uniform(-20, 20, (self.range*2+1, self.range*2+1))

        #Constants for the time function
        self.a = np.random.uniform(1, 2.5)
        self.b = np.random.uniform(1, 2.5)
        self.c = np.random.uniform(2, 4)

        self.func_grid = self.f(self.vals[:, 0], self.vals[:, 1])
        self.max_ = self.max()
        self.min_ = self.min()
        
        self.actions_done = []
        self.previous_closeness_to_max = 0

        self.grid = np.zeros((3, self.resolution, self.resolution), dtype = np.uint8)

        self.check_init_points()

        return self.get_state()

    def get_state(self):
        return ((self.grid.transpose(1, 2, 0)/np.linalg.norm(self.grid))*255).astype(np.uint8)

    def valid_moves(self):
        return 1 - self.grid[1]

    def find_indices(self, x, y):
        #TODO optimize this, only need to call digitize 4 times, not 8
        indicies = []
        for x_bin in [self.x_bins_left, self.x_bins_right]:
            for y_bin in [self.y_bins_left, self.y_bins_right]:
                x_ind, y_ind = np.digitize(x, x_bin), np.digitize(y, y_bin)
                if x_ind > 0 and y_ind > 0 and x_ind <= self.resolution and y_ind <= self.resolution:
                    indicies.append((x_ind-1, y_ind-1))

        if len(indicies) == 0:
            print(x, y)
        return indicies

    def step(self, action, init=False) -> Tuple[np.ndarray, float, bool]:
        """Completes the given action and returns the new map"""
        x, y = action[0], action[1]

        indicies = self.find_indices(x, y)
        time = self.t(x, y)
        for x_ind, y_ind in indicies:
            #print(x_ind, y_ind)
            self.grid[0, x_ind, y_ind] = max(self.f(x, y), self.grid[0, x_ind, y_ind])
            
            self.grid[1, x_ind, y_ind] = 1
            self.grid[2, x_ind, y_ind] = max(time, self.grid[2, x_ind, y_ind])

        
        self.steps += 1
        self.time += time

        reward = self.get_reward()

        self.previous_closeness_to_max = self.get_closeness_to_max()

        self.actions_done.append(((x-self.x_min)*(self.resolution-1)/(self.x_max-self.x_min), (y-self.y_min)*(self.resolution-1)/(self.y_max-self.y_min)))
        if not init:
            self.all_actions.append((x, y))

        pred_max, true_max = self.get_pred_true_max()
        return self.get_state(), reward, bool(self.time > self.T), {"pred_max": pred_max, "true_max": true_max}


    def display_axis(self, idx, axs, fig, data, title, invert = True):
        im = axs[idx].imshow(data)
        fig.colorbar(im, ax=axs[idx])
        if invert:
            axs[idx].invert_yaxis()
        axs[idx].set_title(title)

    def render(self, mode='human', close=False):
        #im = self.get_state()

        if mode == "human":
            if close:
                plt.cla()
                plt.close()
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))
            self.display_axis(0, axs, fig, self.func_grid.reshape(self.resolution, self.resolution), "Function")
            self.display_axis(1, axs, fig, self.t(self.vals[:, 0], self.vals[:, 1]).reshape(self.resolution, self.resolution), "Time")
            self.display_axis(2, axs, fig, self.grid[0], "Features for PPO", invert = True)

            max_coords = np.argmax(self.func_grid)
            y_max, x_max = divmod(max_coords, self.resolution)

            for y, x in self.actions_done[:self.num_init_points]:
                axs[0].scatter(x, y, c = "red", linewidths=7)
                axs[1].scatter(x, y, c = "red", linewidths=7)

            for y, x in self.actions_done[self.num_init_points:-1]:
                axs[0].scatter(x, y, c = "blue", linewidths=7)
                axs[1].scatter(x, y, c = "blue", linewidths=7)

            y, x = self.actions_done[-1]
            print("Last action idx pos", x, y)
            axs[0].scatter(x, y, c = "green", linewidths=7)
            axs[1].scatter(x, y, c = "green", linewidths=7)

            axs[0].scatter(x_max, y_max, c = "black", linewidths = 5)
            axs[1].scatter(x_max, y_max, c = "black", linewidths = 5)

            plt.axis("off")
            plt.show()

        else:
            actions = list(zip(*self.all_actions[-20000:]))
            fig, ax = plt.subplots()
            img = ax.hist2d(actions[0], actions[1], bins = [np.arange(self.x_min, self.x_max, (self.x_max-self.x_min)/self.resolution),
            np.arange(self.y_min, self.y_max, (self.y_max-self.y_min)/self.resolution)], norm=mpl.colors.LogNorm())
            #fig.colorbar(img, ax=ax)
            fig.title("Action distribution")

            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            return data.reshape((int(h), int(w), -1))