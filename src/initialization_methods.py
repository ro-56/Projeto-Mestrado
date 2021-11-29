import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from classes import individual

def make_kmeans_individual(data_points, number_of_clusters):
    return individual(__solve_kmeans(data_points, number_of_clusters))

    
def make_kmedoids_individual(data_points, number_of_clusters):
    return individual(__solve_kmedoids(data_points, number_of_clusters))


def __solve_kmeans(data_points, number_of_clusters):
    X = np.array(data_points)
    kmeans = KMeans(n_clusters=number_of_clusters).fit(X)
    return kmeans.labels_.tolist()


def __solve_kmedoids(data_points, number_of_clusters):
    X = np.array(data_points)
    kmeans = KMedoids(n_clusters=number_of_clusters).fit(X)
    return kmeans.labels_.tolist()