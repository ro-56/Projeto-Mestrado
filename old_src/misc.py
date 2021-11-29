from time import perf_counter
import math
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    printProgressBar(0)
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()

# decorators
# -------------------
def timer(func):
    def wrapper(*args, **kwargs):
        startTime = perf_counter()
        ret = func(*args, **kwargs)
        delta = perf_counter() - startTime
        print(func.__name__, "elapsed time: ", delta)
        return ret
    return wrapper


# -------------------

def solveKmeans(points, nClusters):
    X = np.array(points)
    kmeans = KMeans(n_clusters=nClusters).fit(X)
    return kmeans.labels_.tolist()

def solveKmedoids(points, nClusters):
    X = np.array(points)
    kmeans = KMedoids(n_clusters=nClusters).fit(X)
    return kmeans.labels_.tolist()
    
def getDistanceMatrix(points, dist = ""):

    validDistances = {""}
    if dist not in validDistances:
        raise ValueError("results: status must be one of %r." % validDistances)
    
    dist = [[0 for _ in range(len(points))] for _ in range(len(points))]

    for i in range(len(points)):
        for j in range(len(points)):
            if (i == j):
                dist[i][j] = 0
                continue
            if (j < i):
                continue
            d = 0.0
            for k in range(len(points[0])):
                d += (points[i][k] - points[j][k])**2
            d = math.sqrt(d)
            dist[i][j] = d
            dist[j][i] = d

    return dist


# Auxiliar Functions
# --------------------

def preparaBD(arquivo):
    df = pd.read_csv(arquivo)
    Cols = len(df.columns)
    df2 = df.iloc[:,0:Cols-1]
    df3 = df.iloc[:,Cols-1:Cols]
    Cols = len(df2.columns)
    N = len(df2)
    return df2, N, Cols, df3

def getNeighbors(df):
    df2 = df.values.tolist()
    neigh = NearestNeighbors(n_neighbors=len(df))
    neigh.fit(df2)
    return neigh.kneighbors(df2, return_distance=False)


# graphs
# --------------------

def normalize_data(lst):
    arr = np.asarray(lst)
    normArr = []
    for i in range(len(arr[0])):
        aux = []
        aux = (arr[:,i] - np.min(arr[:,i])) / (np.max(arr[:,i]) - np.min(arr[:,i]))
        normArr.append(aux)
    normArr = np.stack(normArr)
    return normArr.transpose().tolist()