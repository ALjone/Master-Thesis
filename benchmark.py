from game import BlackBox
import numpy as np
from tqdm import tqdm


def benchmark(sims = 10000):
    game = BlackBox()
    #game.reset()
    R = 0
    t = 0

    for i in tqdm(range(sims)):
        game.reset()

        done = False
        while not done:
            _, r, done, _ = game.step((np.random.uniform(game.x_min, game.x_max), np.random.uniform(game.y_min, game.y_max)))
            R += r
            t += 1

    print(f"Benchmark:\n\tAvg reward {round(R/sims, 3)}, Avg steps {round(t/sims, 3)}")

if __name__ == "__main__":
    benchmark()