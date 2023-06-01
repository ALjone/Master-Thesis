import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to rotate x and y coordinates
def rotate(x, y, angles):
    x_rot = np.cos(angles) * x - np.sin(angles) * y
    y_rot = np.sin(angles) * x + np.cos(angles) * y
    return x_rot, y_rot

# generate 10 random pairs of A and B
params = np.random.rand(10, 2)
params[:, 0] = params[:, 0]*5+5
params[:, 1] = params[:, 1]*100

# generate x and y values
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)

# create a meshgrid
X, Y = np.meshgrid(x, y)

# add an extra dimension to X and Y for the rotations
X = np.repeat(X[None, :, :], 10, axis=0)
Y = np.repeat(Y[None, :, :], 10, axis=0)

print(X.shape)

# get the A and B values
A = params[:, 0][:, None, None]
B = params[:, 1][:, None, None]

print(A.shape)

# generate 10 random rotation angles
angles = (np.pi*np.random.rand(10))[:, None, None]

# rotate X and Y coordinates
X_rot, Y_rot = rotate(X, Y, angles)

# calculate Z values (heights) for all functions at once
Z = np.log10(1 + 1 / ((A - X_rot) ** 2 + B * (Y_rot - X_rot**2) ** 2))

#Z = np.log2(Z)
# plot the functions
fig = plt.figure(figsize=(12, 8))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(Z[i], cmap='viridis')
    ax.set_title('A = {:.2f}, B = {:.2f}, angle = {:.2f}'.format(params[i, 0], params[i, 1], angles[i, 0, 0]))

plt.tight_layout()
plt.show() 