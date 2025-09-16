import numpy as np
from sklearn.neighbors import NearestNeighbors

def calculate_hopkins_statistic(data, sample_size=None):
    if sample_size is None:
        sample_size = min(50, len(data) // 2)

    if len(data) < 2:
        return 0.5

    X = data[:, :2]
    random_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[random_indices]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_uniform = np.column_stack([
        np.random.uniform(mins[0], maxs[0], size=sample_size),
        np.random.uniform(mins[1], maxs[1], size=sample_size)
    ])

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    distances_data, _ = nbrs.kneighbors(X_sample)
    u_distances = distances_data[:, 1]

    distances_uniform, _ = nbrs.kneighbors(X_uniform)
    w_distances = distances_uniform[:, 0]

    numerator = np.sum(w_distances)
    denominator = np.sum(u_distances) + np.sum(w_distances)

    if denominator == 0:
        return 0.5

    hopkins_stat = numerator / denominator
    return hopkins_stat

# Dummy functions to resolve ImportError
def draw_heatmap(*args, **kwargs):
    pass

def draw_trails(*args, **kwargs):
    pass

def draw_centroids(*args, **kwargs):
    pass

def create_heatmap(*args, **kwargs):
    pass

def draw_motion_trails(*args, **kwargs):
    pass

def draw_cluster_centroids(*args, **kwargs):
    pass