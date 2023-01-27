import numpy as np

class TileCoder:
    def __init__(self, resolution, domain, n_tiles) -> None:
        x_min = domain[0]
        x_max = domain[1]
        y_min = domain[2]
        y_max = domain[3]

        self.resolution = resolution

        #Need there to fit half the resolution, plus an additional half box on each side
        x_bin_size = (x_max-x_min)/((resolution//2-1)+0.5)
        y_bin_size = (y_max-y_min)/((resolution//2-1)+0.5)
        
        #Create the bins used for the overlapping boxes when turning continous point into discrete label
        self.x_bins_left = np.linspace(x_min, x_max-x_bin_size/2, resolution//2, dtype=np.float64)
        self.x_bins_right = np.linspace(x_min+x_bin_size/2, x_max, resolution//2, dtype=np.float64)

        print(len(self.x_bins_left))
        print(len(self.x_bins_right))
        print()


        self.y_bins_left = np.linspace(y_min, y_max-y_bin_size/2, resolution//2, dtype=np.float64)
        self.y_bins_right = np.linspace(y_min+y_bin_size/2, y_max, resolution//2, dtype=np.float64)
       
        print(y_bin_size/2)
        print(self.y_bins_left+y_bin_size/2)
        print(self.y_bins_right)
        print(self.y_bins_left[1]-self.y_bins_right[0])
        print()

    def __getitem__(self, x):
        x, y = x
        indices = []

        left_x_ind = np.digitize(x, self.x_bins_left, right = False)*2
        right_x_ind = np.digitize(x, self.x_bins_right, right = False)*2+1

        left_y_ind = np.digitize(y, self.y_bins_left, right = False)*2
        right_y_ind = np.digitize(y, self.y_bins_right, right = False)*2+1

        print(left_x_ind, right_x_ind, left_y_ind, right_y_ind)

        for x_ind in [left_x_ind, right_x_ind]:
            for y_ind in [left_y_ind, right_y_ind]:
                if x_ind > 0 and y_ind > 0 and x_ind <= self.resolution and y_ind <= self.resolution:
                    indices.append((x_ind-1, y_ind-1))

        print("Number of squares:", len(indices))
        print("Indicies:", indices)
        print("x, y", x, y)
        print()
        return indices


for resolution in [40]:
    coder = TileCoder(resolution, [0, 10, 0, 10], 0)

coder[10, 10]

coder[0, 0]
