import numpy as np
import torch

class TileCoder:
    def __init__(self, resolution, domain) -> None:
        x_min = domain[0]
        x_max = domain[1]
        y_min = domain[0]
        y_max = domain[1]
        z_min = domain[0]
        z_max = domain[1]

        """self.resolution = resolution

        #Need there to fit half the resolution, plus an additional half box on each side
        x_bin_size = (x_max-x_min)/((resolution//2-1)+0.5)
        y_bin_size = (y_max-y_min)/((resolution//2-1)+0.5)
        z_bin_size = (z_max-z_min)/((resolution//2-1)+0.5)
        
        #Create the bins used for the overlapping boxes when turning continous point into discrete label
        self.x_bins_left = np.linspace(x_min, x_max-x_bin_size/2, resolution//2, dtype=np.float16)
        self.x_bins_right = np.linspace(x_min+x_bin_size/2, x_max, resolution//2, dtype=np.float16)

        self.y_bins_left = np.linspace(y_min, y_max-y_bin_size/2, resolution//2, dtype=np.float64)
        self.y_bins_right = np.linspace(y_min+y_bin_size/2, y_max, resolution//2, dtype=np.float64)

        self.z_bins_left = np.linspace(z_min, z_max-z_bin_size/2, resolution//2, dtype=np.float64)
        self.z_bins_right = np.linspace(z_min+z_bin_size/2, z_max, resolution//2, dtype=np.float64)"""

        #NO TILE CODING WITH THIS:

        self.x_bins = torch.linspace(x_min, x_max, resolution).to(torch.device("cuda"))
        self.y_bins = torch.linspace(y_min, y_max, resolution).to(torch.device("cuda"))
        self.z_bins = torch.linspace(z_min, z_max, resolution).to(torch.device("cuda"))

       

    def __getitem__(self, x):
        #TODO: Needs to work with batches?
        x, y, z = x
        x = torch.bucketize(x, self.x_bins)
        y = torch.bucketize(y, self.y_bins)
        z = torch.bucketize(z, self.z_bins)
        return (x, y, z)

        #NOTE: This below works, just need to add z

        left_x_ind = np.digitize(x, self.x_bins_left, right = True)*2
        right_x_ind = np.digitize(x, self.x_bins_right, right = False)*2+1

        left_y_ind = np.digitize(y, self.y_bins_left, right = True)*2
        right_y_ind = np.digitize(y, self.y_bins_right, right = False)*2+1



        #print(left_x_ind, right_x_ind, left_y_ind, right_y_ind)

        for x_ind in [left_x_ind, right_x_ind]:
            for y_ind in [left_y_ind, right_y_ind]:
                if x_ind > 0 and y_ind > 0 and x_ind <= self.resolution and y_ind <= self.resolution:
                    indices.append((x_ind-1, y_ind-1))

        #print("Number of squares:", len(indices))
        #print("Indicies:", indices)
        #print("x, y", x, y)
        #print()
        return indices

