import matplotlib.pyplot as plt
from game import BlackBox
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
import matplotlib as mpl
#import benchmark


model = None#PPO("CnnPolicy", BlackBox())#PPO.load("Best")
model = PPO.load("Best")
game = BlackBox()
#game.reset()
R = 0
t = 0
sims = 5000

x_actions = []
y_actions = []

avg = np.zeros(game.grid.shape[1:])

for i in tqdm(range(sims)):
    s = game.reset()
    avg += game.func_grid.reshape(game.resolution, game.resolution)
    done = False
    #continue
    while not done:
        if model is not None:
            action, _states = model.predict(s)
        else:
            action = (np.random.uniform(game.x_min, game.x_max), np.random.uniform(game.y_min, game.y_max))
        x_actions.append(action[0])
        y_actions.append(action[1])

        s, r, done, info = game.step(action)

        R += r
        t += 1

avg /=sims

im = plt.imshow(avg)
plt.colorbar(im)
plt.title("Avg of all functions checked during test")
plt.show()

resolution = 50
fig, ax = plt.subplots()
h = ax.hist2d(x_actions, y_actions, bins = [np.arange(game.x_min, game.x_max, (game.x_max-game.x_min)/resolution), np.arange(game.y_min, game.y_max, (game.y_max-game.y_min)/resolution)])
fig.colorbar(h[3], ax=ax)
plt.title("Heat map of actions taken")
plt.show()


fig, ax = plt.subplots()
h = ax.hist2d(x_actions, y_actions, bins = [np.arange(game.x_min, game.x_max, (game.x_max-game.x_min)/resolution), np.arange(game.y_min, game.y_max, (game.y_max-game.y_min)/resolution)], norm=mpl.colors.LogNorm())
fig.colorbar(h[3], ax=ax)
plt.title("Log normalized heat map of actions taken")
plt.show()


print(f"Avg reward {round(R/sims, 3)}, Avg steps {round(t/sims, 3)}")