import numpy as np
# Define the three original vectors
vector1 = np.arange(0, 40)
vector2 = np.arange(0, 40)
vector3 = np.arange(0, 40)

# Create a meshgrid of indices
idx1, idx2, idx3 = np.meshgrid(np.arange(len(vector1)),
                               np.arange(len(vector2)),
                               np.arange(len(vector3)),
                               indexing='ij')

# Stack the indices and use them to index the original vectors
result = np.column_stack((vector1[idx1.flatten()],
                          vector2[idx2.flatten()],
                          vector3[idx3.flatten()]))

# Reshape the result to the desired shape
result = result.reshape(-1, 3)

# Print the result
print(result)
print(result.shape)