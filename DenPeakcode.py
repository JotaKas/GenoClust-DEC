# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import metrics

def ChooseCenter_N_Cluster(dc_percent, X):
    """
    Identify potential cluster centers based on local density and distance from points with higher density.
    """
    dis_matrix = euclidean_distances(X, X)
    avg_dis = np.average(dis_matrix)
    dc = dc_percent * avg_dis

    dis_matrix1 = dis_matrix / dc
    dis_matrix1 = np.multiply(dis_matrix1, dis_matrix1)
    dis_matrix1 = np.exp(-dis_matrix1)
    rho = dis_matrix1.sum(axis=1) - 1

    rho_sorted = sorted([(rho_val, i) for i, rho_val in enumerate(rho)], reverse=True)
    delta = np.zeros_like(rho)
    nneigh = np.zeros(len(X), dtype=int)

    for i in range(len(rho)):
        if i == rho_sorted[0][1]:
            delta[i] = dis_matrix[i].max()
        else:
            delta[i] = np.min([dis_matrix[i][j] for _, j in rho_sorted[:i]])

    cl = np.full(len(X), -1)
    center_num = 0
    for i, (rho_val, idx) in enumerate(rho_sorted):
        if delta[idx] > dc:
            cl[idx] = center_num
            center_num += 1
        else:
            distances = dis_matrix[idx][cl >= 0]
            if len(distances) > 0:
                cl[idx] = cl[np.argmin(distances)]

    return dc, cl, rho, delta, center_num

def Corepoints_N_Merge(dc, cl, rho, X):
    """
    Refine clusters by merging based on core points and border points.
    """
    cluster_labels = np.unique(cl)
    for i in cluster_labels:
        core_points = rho[cl == i] > np.percentile(rho, 50)  # Example criterion for core points
        if not np.any(core_points):
            continue
        core_indices = np.where((cl == i) & core_points)[0]
        for core_idx in core_indices:
            for j in cluster_labels:
                if i == j:
                    continue
                other_core_indices = np.where((cl == j) & core_points)[0]
                for other_core_idx in other_core_indices:
                    if euclidean_distances(X[core_idx].reshape(1, -1), X[other_core_idx].reshape(1, -1)) < dc:
                        cl[cl == j] = i  # Merge cluster j into cluster i
                        break

    # Recalculate the number of centers after merging
    center_num = len(np.unique(cl))

    return cl, center_num

def DenPeakCluster(X, dc_percent=0.1):
    """
    Perform Density Peaks clustering with refinement on encoded genomic features.
    """
    dc, cl, rho, delta, center_num = ChooseCenter_N_Cluster(dc_percent, X)
    cl, center_num = Corepoints_N_Merge(dc, cl, rho, X)

    print(f"Number of clusters after refinement: {center_num}")
    
    # Optional: Evaluation and visualization steps here

    return cl, center_num

# Example usage:
# encoded_features = ... # Your encoded genomic data here
# cl, center_num = DenPeakCluster(encoded_features)
