import numpy as np
from scipy.spatial import distance_matrix

def k_means(X, k):
    nrow = X.shape[0]
    ncol = X.shape[1]

    initial_centroids = np.random.choice(nrow, k, replace=False)

    centroids = X[initial_centroids]
    centroids_old = np.zeros(k, ncol)

    cluster_assignments = np.zeros(nrow)
    while centroids_old != centroids:
        centroids_old = centroids.copy()

        #compute distances between data points and centroids
        dist_matrix = distance_matrix(X, centroids, p=2)

        for i in np.arange(nrow):
            #Find closest centroid
            d = dist_matrix[i]
            closest_centroid = d.index(np.min(d))

            #Associate data point with closest centroid
            cluster_assignments[i] = closest_centroid

        #recompute centroids
        for j in np.arange(k):
            temp = X[cluster_assignments == j]
            centroids[j] = np.apply_along_axis(np.mean, axis=1, arr=temp)

    return (centroids, cluster_assignments)

