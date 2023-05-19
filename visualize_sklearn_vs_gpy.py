from tqdm import tqdm
from Old.GP import GP as sklearn_GP
from sklearn.gaussian_process.kernels import RBF
from GPY import GP as gpy_GP
import numpy as np
import torch
import matplotlib.pyplot as plt
from random_function import RandomFunction
from scipy.stats import norm
import imageio
import numpy as np

def create_gif(frames, filename, duration=0.1):
    # Convert each frame (array) to uint8 format
    frames = [np.uint8(frame) for frame in frames]

    # Save the frames as a GIF using imageio
    imageio.mimsave(filename, frames, duration=duration)

# Example usage:
# Assuming you have a list called 'frames' containing the arrays
# and you want to save the GIF as 'animation.gif'



def EI(u, std, biggest, e = 0.01):
    if std <= 0:
        print("std under 0")
        return 0
    Z = (u-biggest-e)/std
    return (u-biggest-e)*norm.cdf(Z)+std*norm.pdf(Z)

def get_next_x_y(sklearn_gp: sklearn_GP, gpy_gp: gpy_GP, x: np.ndarray, y: np.ndarray, matrix: np.ndarray):
    
    next_point = np.array(gpy_gp.get_next_point(EI, np.max(y)))
    sklearn_gp.get_next_point()

    next_value = matrix[0, next_point[0], next_point[1]]

    next_point = next_point/30

    sklearn_gp.update_points(next_point, next_value)
    x = np.concatenate((x, np.expand_dims(next_point, 0)))
    y = np.concatenate((y, np.expand_dims(next_value, 0)))
    gpy_gp.get_mean_std(torch.tensor(x).to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")),
                         torch.tensor(y).to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")), torch.arange(start = 0, end = 2).to(torch.device("cuda")))

    return x, y
    

def run():
    matrix = RandomFunction((0, 1), 30, 2, dims = 2).matrix.cpu().numpy()

    x = np.array([[0, 0], [0.5, 0.7], [0.3, 0.3]])
    y = np.array([matrix[0, int(x_[0]*30), int(x_[1]*30)] for x_ in x]).squeeze()

    sklearn_gp = sklearn_GP([RBF()*1], EI, ((0, 1), (0, 1)), 30, ("One", "Two"), checked_points=x, values_found=y)
    gpy_gp =  gpy_GP(None, 2, (0, 1), 30, dims = 2, verbose=0, training_iters=200, approximate=False)

    gpy_gp.get_mean_std(torch.tensor(x).to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")),
                         torch.tensor(y).to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")), torch.arange(start = 0, end = 2).to(torch.device("cuda")))

    sklearn_imgs = []
    gpy_imgs = []
    for i in tqdm(range(20)):
        x, y = get_next_x_y(sklearn_gp, gpy_gp, x, y, matrix)
        sklearn_imgs.append(sklearn_gp.render())
        gpy_imgs.append(gpy_gp.render())
    print("Max found:", round(np.max(y).item(), 3))
    create_gif(sklearn_imgs, 'sklearn.gif', duration=1)
    create_gif(gpy_imgs, 'gpy.gif', duration=1)
    

run()