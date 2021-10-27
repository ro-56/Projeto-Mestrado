from classes import indiv

class edge():
    fromNode: int
    toNode: int
    dist: float

    def __init__(self, fromNode: int, toNode: int, dist: float ):
        self.fromNode = fromNode
        self.toNode = toNode
        self.dist = dist

def kruscalGenIndiv(distMatrix, Cmin: int, Cmax: int):
    edge_list = makeListFromDistMatrix(distMatrix)
    edge_list.sort(key=lambda x: x.dist, reverse=False)

    roots = list(range(len(distMatrix)))
    numTrees = len(distMatrix)

    ret = []

    while numTrees > Cmin:
        currentEdge = edge_list.pop(0)

        if (roots[currentEdge.fromNode] == roots[currentEdge.toNode]):
            continue
        
        for i in range(len(roots)):
            if roots[i] == max(roots[currentEdge.fromNode], roots[currentEdge.toNode]):
                roots[i] = min(roots[currentEdge.fromNode], roots[currentEdge.toNode])

        numTrees -= 1
        
        if numTrees <= Cmax:
            ret.append(indiv(roots))

    return ret


def makeListFromDistMatrix(distMatrix: list[list[float]]) -> list[edge]:
    ret = []

    for i in range(len(distMatrix)):
        for j in range(len(distMatrix)):
            if (j <= i):
                continue
            ret.append(edge(i, j, distMatrix[i][j]))

    return ret