from tqdm import tqdm
from sklearn_GP import GP as sklearn_GP
from sklearn.gaussian_process.kernels import RBF
from env.GPY import GP as gpy_GP
import numpy as np
import torch
import matplotlib.pyplot as plt
from random_functions.convex_functions import RandomFunction
from scipy.stats import norm
import imageio
import numpy as np
from acquisition_functions import EI

def create_gif(frames, filename, duration=0.1):
    # Convert each frame (array) to uint8 format
    frames = [np.uint8(frame) for frame in frames]

    # Save the frames as a GIF using imageio
    imageio.mimsave(filename, frames, duration=duration)

def pad_sublists(list_of_lists, expand_size):
    idx = (0, 1)
    padded_lists = []
    for i in idx:
        sublist = list_of_lists[i]
        sublist_len = len(sublist)
        num_copies = expand_size // sublist_len
        padding_len = expand_size % sublist_len
        padded_sublist = sublist * num_copies + sublist[:padding_len]
        padded_lists.append(torch.stack(padded_sublist))

    return torch.stack(padded_lists)


def get_next_x_y(sklearn_gp: sklearn_GP, gpy_gp: gpy_GP, x: torch.tensor, y: torch.tensor, matrix: torch.tensor, i: int):
    
    next_point = torch.tensor(gpy_gp.get_next_point(EI, torch.max(y)))
    sklearn_gp.get_next_point()

    next_value = matrix[0, next_point[0], next_point[1]]

    next_point = next_point/30

    sklearn_gp.update_points(next_point.numpy(), next_value.numpy())
    x[i] = next_point
    y[i] = next_value
    gpy_gp.get_mean_std(x.to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")),
                        y.to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")), torch.arange(start = 0, end = 2).to(torch.device("cuda")))

    return x, y
    

def run():
    print("Warning, this doesn't function the same as batched env. Rewrite to work like batched env by making it use the list of list stuff")
    matrix = RandomFunction().matrix.cpu()

    _x = np.array([[0.5, 0.7], [0.3, 0.3]])
    _y = np.array([matrix[0, int(x_[0]*30), int(x_[1]*30)] for x_ in _x]).squeeze()

    sklearn_gp = sklearn_GP([RBF()*1], EI, ((0, 1), (0, 1)), 30, ("One", "Two"), checked_points=_x, values_found=_y)
    gpy_gp =  gpy_GP()

    x = torch.tensor(np.tile(_x[0], (50, 1)))
    y = torch.tensor(_y[0].repeat(50))
    x[1] = torch.tensor(_x[1])
    y[1] = torch.tensor(_y[1])

    gpy_gp.get_mean_std(x.to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")),
                        y.to(torch.float32).unsqueeze(0).repeat_interleave(2, 0).to(torch.device("cuda")), torch.arange(start = 0, end = 2).to(torch.device("cuda")))

    sklearn_imgs = []
    sklearn_gp.get_next_point()
    sklearn_imgs.append(sklearn_gp.render())
    plt.imshow(sklearn_imgs[-1])
    plt.show()
    gpy_imgs = []
    gpy_imgs.append(gpy_gp.render())
    plt.imshow(gpy_imgs[-1])
    plt.show()
    i = 2
    for _ in tqdm(range(20)):
        x, y = get_next_x_y(sklearn_gp, gpy_gp, x, y, matrix, i)
        sklearn_imgs.append(sklearn_gp.render())
        gpy_imgs.append(gpy_gp.render())
        i += 1
    print(y)
    print("Max found:", round(torch.max(y).item(), 3))
    create_gif(sklearn_imgs, 'sklearn.gif', duration=1)
    create_gif(gpy_imgs, 'gpy.gif', duration=1)
    

run()