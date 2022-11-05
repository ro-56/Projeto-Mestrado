from .readData import DataSource


class Individual:
    _maxNumCluster: int
    _numProxPoints: int
    _size: int

    genotype: list[int]
    fitness: list[float]

    def updateFitness01(self, numAttributes:int, dfValues):
        centroidList, pointsIdxInCluster = self.getCentroids(numAttributes, dfValues)
        pointsInCluster = [[dfValues[idxPoint] for idxPoint in pointsIdxInCluster[idxCluster]] for idxCluster in range(self.maxNumCluster)]
        finalsum = 0
        for idxCluster in range(self.maxNumCluster):
            if not pointsInCluster[idxCluster]:
                continue
            finalsum += np.linalg.norm(centroidList[idxCluster] - np.array(pointsInCluster[idxCluster])) 
        self.fitness01 = finalsum

    def updateFitness02(self, neighborMatrix):
        totalSum = 0
        for idxData in range(self.size):
            for idxNearPoint in range(1,self.numProxPoints + 1):
                if self.data[neighborMatrix[idxData][idxNearPoint]] == self.data[idxData]:
                    continue
                totalSum += 1/float(idxNearPoint+1)
        self.fitness02 = totalSum

    def updateAllFitness(self, ds: DataSource):
        self.updateFitness01(numAttributes, dfValues)
        self.updateFitness02(neighborMatrix)

    def reset_fitness(self):
        self.fitness = 0

class Population:
    _size: int

    members: list[Individual]

    def __init__(self, members: list[Individual]=None):
        if members:
            self.members = members
        else:
            self.members = []

    def get_population_size(self):
        return len(self.members)
    
    def updateAllFitness(self, ds: DataSource):
        for indiv in self.members:
            indiv.updateAllFitness(ds)
    
    def addMember(self, indiv:Individual):
        self.members.append(indiv)


