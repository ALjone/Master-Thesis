import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
from tqdm import tqdm
from itertools import product
import math
import matplotlib.pyplot as plt

def cross_validate_2d(clf, X, y, kernel):
    score = 0
    for i in range(len(X)):
        x_data = np.delete(X, i, 0)
        y_data = np.delete(y, i)
        clf.fit(x_data, y_data)
        score += np.abs(clf.predict(X[i].reshape(1, -1))-y[i].reshape(1, -1))
        
        clf.kernel_ = kernel

    clf.fit(X, y)
    return -score, clf

#TODO: Rewrite to run on GPU with gpytorch
#https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html

class GP:
    def __init__(self, kernels, acquisition, data_range, resolution, names, verbose = 0, use_tqdm = True, checked_points = None, values_found = None) -> None:

        #TODO: REWRITE TO WORK FROM 0-1 OR SOMETHING
        self.kernels = kernels
        self.acquisition = acquisition
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.resolution = resolution
        #Distance between this point and the next
        self.distances = []
        self.names = names
        self.bounds = data_range

        self.original_dims = np.zeros((len(data_range), resolution), dtype=np.float64)
        self.transformed_dims = np.zeros((len(data_range), resolution), dtype=np.float64)
        for i, r in enumerate(data_range):
            self.original_dims[i] = np.linspace(r[0], r[1], resolution, dtype=np.float64)
            self.transformed_dims[i] = np.linspace(0, 10, resolution, dtype=np.float64)

        self.checked_points = np.load("checked_points.npy") if checked_points is None else checked_points
        self.checked_points_transformed = self._transform_actions(self.checked_points)
        self.values_found = np.load("values_found.npy") if values_found is None else values_found


        self.initial_points = self.checked_points.copy()
        self.biggest = np.max(self.values_found)
        self.biggest_coords = self.checked_points[np.argmax(self.values_found)]
        if self.verbose > 0:
            print("Biggest:", self.biggest)


    def get_next_point(self):
        warnings.filterwarnings("ignore")
        scores = []
        for _, kernel in enumerate(self.kernels):
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=None, n_restarts_optimizer=2, normalize_y = False, alpha=1e-5)
            score, gpr = cross_validate_2d(gpr, self.checked_points_transformed, self.values_found, kernel)

            scores.append((score, gpr)) 
        scores = sorted(scores, key=lambda tup: tup[0])

        #Print best scores
        if self.verbose > 0:
            for s in scores:
                print("Score:", round(s[0][0][0], 3), "Kernel:", s[1].kernel_)
        
        #Save best GPR
        self.gpr: GaussianProcessRegressor = scores[-1][1]
        warnings.simplefilter('always')

        self._predict_matrix()
        best_point = None
        best_ei = -np.inf
        for i in range(self.mean.shape[0]):
            for j in range(self.mean.shape[1]):
                ei = self.acquisition(self.mean[i,j], self.std[i,j], self.biggest)
                if ei > best_ei:
                    best_point = (i,j)
                    best_ei = ei
        return best_point

    def _predict_matrix(self):


        self.mean = np.zeros([self.resolution for _ in range(self.transformed_dims.shape[0])])
        self.std = np.zeros([self.resolution for _ in range(self.transformed_dims.shape[0])])

        points = product(range(self.resolution), repeat = self.transformed_dims.shape[0])
        for ijk_point in tqdm(points, leave = False, disable=~self.use_tqdm, desc = "Predicting mean matrix"):      
            point = np.array([self.transformed_dims[i, p] for i, p in enumerate(ijk_point)]).reshape(1, -1)
            mean, std = self.gpr.predict(point, return_std=True)

            self.mean[ijk_point] = mean
            self.std[ijk_point] = std

    def update_points(self, check_coords, value):
        self.checked_points = np.append(self.checked_points, np.array(check_coords, dtype=np.float64).reshape(1, self.original_dims.shape[0]), axis = 0)
        self.checked_points_transformed = np.append(self.checked_points_transformed, self._transform_actions(np.array(check_coords, dtype=np.float64)).reshape(1, self.original_dims.shape[0]), axis = 0)
        self.values_found = np.append(self.values_found, value.astype(np.float64)) #This is the same as "simulating" the result 
        self.biggest = np.max(self.values_found)
        self.biggest_coords = self.checked_points[np.argmax(self.values_found)]
        self.distance_from_last_point = np.linalg.norm(self.checked_points[-1]-self.checked_points[-2])
        self.distances.append(self.distance_from_last_point)
        self.save()

    def _transform_actions(self, vals):
        if len(vals.shape) == 1:
            vals = np.expand_dims(vals, 0)
        new_vals = np.zeros(vals.shape, dtype=np.float64)
        for j in range(new_vals.shape[0]):
            for i, (val, bound) in enumerate(zip(vals[j], self.bounds)):
                new_vals[j, i] = self.__transform_actions(val, bound[0], bound[1])
        return new_vals

    def __transform_actions(self, val, old_lower, old_upper):
        upper_bound, lower_bound = 10, 0
        OldRange = old_upper - old_lower
        NewRange = (upper_bound - lower_bound)
        return ((((val - old_lower) * NewRange) / OldRange) + lower_bound)

    def get_info(self):
        def signif(x, digits=6):
            if x == 0 or not math.isfinite(x):
                return x
            digits -= math.ceil(math.log10(abs(x)))
            return round(x, digits)
        round_val = 3
        print(  "\tBest guess position:", "(+", ", ".join([str(signif(coord, round_val)) for coord in self.biggest_coords]) + ")",#f"({round(self.biggest_coords[0], round_val)}, {round(self.biggest_coords[1], round_val)})",
                "\n\tPredicted value last:", signif(self.predicted_best_last, round_val),
                "\n\tTrue value last:", signif(self.values_found[-1], round_val),
                "\n\tMax value found:", signif(self.biggest, round_val), 
                "\n\tLast checked position:", "(" + ", ".join([str(signif(coord, round_val)) for coord in self.check_coords]) + ")",
                "\n\tDistance from last point:", signif(self.distance_from_last_point, round_val),
                "\n\tAverage distance from last point:", signif(np.mean(self.distances), round_val)
          )

    def save(self):
        np.save("checked_points", self.checked_points)    
        np.save("values_found", self.values_found)
        

    def render(self):
        assert self.original_dims.shape[0] < 4
        if self.original_dims.shape[0] == 3:
            return self._render_3d()
        if self.original_dims.shape[0] == 2:
            images = {}
            name_string = f"{self.names[0]} vs {self.names[1]} - Mean"
            images[name_string] = self._render_2d(self.original_dims[0], self.original_dims[1], self.mean, self.names[0], self.names[1], 0, 1)
            return images

    def _render_3d(self):
        images = {}
        for ax in range(self.original_dims.shape[0]):
            ax_1, ax_2 = [(1, 2), (0, 2), (0, 1)][ax]
            name_string = f"{self.names[ax_1]} vs {self.names[ax_2]} - Mean"
            images[name_string] = self._render_2d(self.original_dims[ax_1], self.original_dims[ax_2], np.mean(self.mean, axis=ax), self.names[ax_1], self.names[ax_2], ax_1, ax_2)
        return images

    def _render_2d(self, x, y, points, x_name, y_name, ax_1, ax_2):
        plt.close()
        plt.cla()
        fig, axs = plt.subplots()
        img = axs.contourf(x, y, points, 1000)
        axs.scatter((self.checked_points[:, ax_1]), (self.checked_points[:, ax_2]), c = "red")
        #axs.scatter((self.initial_points[:, ax_1]), (self.initial_points[:, ax_2]), c = "red")
        #axs.scatter(self.biggest_coords[ax_1], self.biggest_coords[ax_2], c = "black")
        axs.set_xlabel(x_name)
        axs.set_ylabel(y_name)
        axs.set_title(f"{x_name} vs {y_name} - Mean")
        plt.colorbar(img, ax = axs)
        plt.show()
        plt.close()
        plt.cla()
        return
        fig.canvas.draw()
        arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.show()
        plt.close('all')
        plt.cla()
        return data
    


if __name__ == "__main__":
    x = np.array([[0, 0], [0.5, 0.7], [0.3, 0.3]])
    y = np.array([0, 3.4, 1.5])
    gp = GP([RBF()*1], None, ((0, 1), (0, 1)), 30, ("One", "Two"), checked_points=x, values_found=y)
    gp.get_next_point()
    print(gp.kernels[0].get_params())
    print(gp.gpr.get_params())
    img = gp.render()
    plt.imshow(img)
    plt.show()