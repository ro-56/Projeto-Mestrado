import copy
import random
import numpy as np

class indiv:
    maxNumCluster:int
    size: int
    numProxPoints: int
    crossoverCuts: int = 5

    def __init__(self, data:list[int]):
        self.resetFitness()
        self.data = data

    def resetFitness(self):
        self.fitness01 = 0
        self.fitness02 = 0
    
    def getCentroids(self, numAttributes:int, dfValues):
        centroidList = [[0 for _ in range(numAttributes)] for _ in range(self.maxNumCluster)]
        pointsInCluster = []
        for idxCluster in range(self.maxNumCluster):
            pointsIdxInCluster = [i for i, point in enumerate(self.data) if point == idxCluster]
            pointsInCluster.append(pointsIdxInCluster)
            for idxAttr in range(numAttributes):
                centroidList[idxCluster][idxAttr] = getAverageValue([dfValues[pointIdx][idxAttr] for pointIdx in pointsIdxInCluster])
        return centroidList, pointsInCluster 

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

    def updateAllFitness(self, neighborMatrix, numAttributes:int, dfValues):
        self.updateFitness01(numAttributes, dfValues)
        self.updateFitness02(neighborMatrix)
    
    def mutation01(self):
        return

    def mutation02(self, neighborMatrix):
        randMutationPoint = random.randint(0,self.size-1)
        for idx in range(self.size):
            if self.data[randMutationPoint] != self.data[neighborMatrix[randMutationPoint][idx]]:
                self.data[randMutationPoint] = self.data[neighborMatrix[randMutationPoint][idx]]
                break
    
    def mutate(self, neighborMatrix):
        if (random.random() < 0.5):
            self.mutation01()
        else:
            self.mutation02(neighborMatrix)
        self.resetFitness()


class population:
    def __init__(self, members = []):
        if len(members):
            self.members = members
        else:
            self.members = []

    def addMember(self, indiv:indiv):
        self.members.append(copy.deepcopy(indiv))

    def mergePopulation(self, otherPop:"population", deepCopy = 0):
        for indiv in otherPop.members:
            if deepCopy:
                self.members.append(copy.deepcopy(indiv))
            else:
                self.members.append(indiv)

    def getSize(self) -> int:
        return len(self.members) 

    def updateAllFitness(self, neighborMatrix, numAttributes:int, dfValues):
        for indiv in self.members:
            indiv.updateAllFitness(neighborMatrix, numAttributes, dfValues)

def getAverageValue(vector):
    if not len(vector):
        return 0
    return sum(vector) / len(vector)