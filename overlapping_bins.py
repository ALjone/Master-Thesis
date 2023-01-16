import numpy as np

def create_overlapping_bins(bin_edges, overlap_percent):
    bin_width = np.diff(bin_edges)
    overlap = bin_width * overlap_percent / 100
    new_bin_left = []
    new_bin_right = []
    for i in range(len(bin_edges) - 1):
        new_bin_right.append(bin_edges[i+1] - overlap[i])
        new_bin_left.append(bin_edges[i])
    return new_bin_left, new_bin_right

x_bins = np.linspace(2, 12, 11)
new_bin_left, new_bin_right = create_overlapping_bins(x_bins, 50)
print(new_bin_left, "\n", new_bin_right)

print(np.digitize(4.5, new_bin_right))

print(np.digitize(4.5 , new_bin_left))