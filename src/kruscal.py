from classes import individual
import math


def make_population_with_kruscal_trees(data_points, min_number_clusters: int, max_number_clusters: int) -> list[individual]:
    edge_list = __get_edge_list(data_points)
    edge_list.sort(key=lambda x: x.dist, reverse=False)

    roots = list(range(len(data_points)))
    num_existing_trees = len(data_points)

    list_of_trees = []

    while num_existing_trees > min_number_clusters:
        currentEdge = edge_list.pop(0)

        if (roots[currentEdge.fromNode] == roots[currentEdge.toNode]):
            continue

        for i in range(len(roots)):
            if roots[i] == max(roots[currentEdge.fromNode], roots[currentEdge.toNode]):
                roots[i] = min(roots[currentEdge.fromNode], roots[currentEdge.toNode])

        num_existing_trees -= 1

        if num_existing_trees <= max_number_clusters:
            list_of_trees.append(individual(roots))

    return list_of_trees


class __edge():
    fromNode: int
    toNode: int
    dist: float

    def __init__(self, fromNode: int, toNode: int, dist: float):
        self.fromNode = fromNode
        self.toNode = toNode
        self.dist = dist


def __get_edge_list(data_points) -> list[__edge]:
    edge_list = []

    dist_matrix = __get_distance_matrix(data_points)

    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if (j <= i):
                continue
            edge_list.append(__edge(i, j, dist_matrix[i][j]))

    return edge_list


def __get_distance_matrix(points, dist = ""):
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