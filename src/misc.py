import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def import_data(filename: str):
    with open(filename, 'r') as file:
        df = pd.read_csv(file)
        Cols = len(df.columns)
        df2 = df.iloc[:,0:Cols-1]
        df3 = df.iloc[:,Cols-1:Cols]
    return df2, df3

def get_neighbors(data_points):
    neigh = NearestNeighbors(n_neighbors=len(data_points))
    neigh.fit(data_points)
    return neigh.kneighbors(data_points, return_distance=False)

# def get_centroids(clusters, data_points):
#     centroid_list = [[0 for _ in range(len(data_points))] for _ in range(len(data_points[0]))]
#     points_in_cluster = []
#     for idxCluster in range(max(clusters)):
#         pointsIdxInCluster = [i for i, point in enumerate(clusters) if point == idxCluster]
#         points_in_cluster.append(pointsIdxInCluster)
#         for idxAttr in range(max(clusters)):
#             centroid_list[idxCluster][idxAttr] = __get_average_value([data_points[pointIdx][idxAttr] for pointIdx in pointsIdxInCluster])
#     return centroid_list, points_in_cluster 

# def __get_average_value(vector):
#     if not len(vector):
#         return 0
#     return sum(vector) / len(vector)

def get_centroid(arr):
    arr = np.array(arr)
    length = arr.shape[0]
    centroid = [np.sum(arr[:, i]) for i in range(arr.shape[1])]
    # sum_x = np.sum(arr[:, 0])
    # sum_y = np.sum(arr[:, 1])
    # return sum_x/length, sum_y/length
    return centroid