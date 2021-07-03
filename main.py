##%%
import copy
import pandas as pd
import numpy as np
import random
# import pylab
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from nsgaII import fast_non_dominated_sort
from nsgaII import crowding_distance
from nsgaII import sort_by_values
from nsgaII import index_of
import gc
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid

# decorators
# -------------------
def timer(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        ret = func(*args, **kwargs)
        delta = time.time() - startTime
        print(func.__name__, "elapsed time: ", delta)
        return ret
    return wrapper

# Auxiliar Functions
# --------------------

def getAverageValue(vector):
    return sum(vector) / len(vector)


# classes
# --------------------
class indiv:

    maxNumCluster = 0
    size = 0
    numProxPoints = 0
    crossoverCuts = 5

    def __init__(self, data = []):
        self.resetFitness()
        if len(data):
            self.data = data
        else:
            self.randomInitialization()

    def resetFitness(self):
        self.fitness01 = 0
        self.fitness02 = 0

    def randomInitialization(self):
        self.data = [random.randint(0,self.maxNumCluster-1) for _ in range(self.size)]
    
    def updateAllFitness(self, neighborMatrix, numAttributes:int, df):
        self.updateFitness01(numAttributes, df)
        self.updateFitness02(neighborMatrix)
    
    def updateFitness01(self, numAttributes:int, df):
        dfValues = df.values.tolist()
        centroidList, pointsIdxInCluster = self.getCentroids(numAttributes, df)
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
                if self.data[neighborMatrix[idxData][idxNearPoint]] == self.data[idxNearPoint]:
                    continue
                totalSum += 1/float(idxNearPoint+1)
        self.fitness02 = totalSum
    
    def getCentroids(self, numAttributes:int, df):
        dfValues = df.values.tolist()
        centroidList = [[0 for _ in range(numAttributes)] for _ in range(self.maxNumCluster)]
        pointsInCluster = []
        for idxCluster in range(self.maxNumCluster):
            pointsIdxInCluster = [i for i, point in enumerate(self.data) if point == idxCluster]
            pointsInCluster.append(pointsIdxInCluster)
            for idxAttr in range(numAttributes):
                centroidList[idxCluster][idxAttr] = getAverageValue([dfValues[pointIdx][idxAttr] for pointIdx in pointsIdxInCluster])
        return centroidList, pointsInCluster 

    def getCrossover(self, otherIndiv:"indiv"):
        cuts = [random.randint(0,self.size-1) for _ in range(self.crossoverCuts)]
        a = [0 for _ in range(self.size)]
        b = [0 for _ in range(self.size)]
        invert = 0
        for i in range(self.size):
            if i in cuts:
                invert = not invert
            if not invert:
                a[i] = self.data[i]
                b[i] = otherIndiv.data[i]
            else:
                a[i] = otherIndiv.data[i]
                b[i] = self.data[i]

        return indiv(a), indiv(b)

    def mutate(self, neighborMatrix):
        if (random.random() < 0.5):
            self.mutation01()
        else:
            self.mutation02(neighborMatrix)
    
    def mutation01(self):
        return

    def mutation02(self, neighborMatrix):
        randMutationPoint = random.randint(0,self.size-1)
        for idx in range(self.size):
            if self.data[randMutationPoint] != self.data[neighborMatrix[randMutationPoint][idx]]:
                self.data[randMutationPoint] = self.data[neighborMatrix[randMutationPoint][idx]]
                break
        self.resetFitness()

class population:

    def __init__(self):
        self.members = []

    def addMember(self, indiv:indiv):
        self.members.append(indiv)

    def mergePopulation(self, otherPop:"population"):
        for indiv in otherPop.members:
            self.members.append(copy.deepcopy(indiv))

    def getSize(self):
        return len(self.members) 

    def updateAllFitness(self, neighborMatrix, numAttributes:int, df):
        for indiv in self.members:
            indiv.updateAllFitness(neighborMatrix, numAttributes, df)

@timer
def main(MAX_GEN = 200, TAM_POP = 100, RODADAS = 1):

    df, N, Cols = preparaBD("diabetes.csv") 

    indiv.maxNumCluster = 15
    indiv.numProxPoints = 10
    indiv.size = N

    matriz3D = []
    for _ in range(RODADAS):
        pontos_function1, pontos_function2 = geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP)
        matriz3D.append([pontos_function1, pontos_function2])
    imprimeRodadas(matriz3D)

    return

def getNeighbors(df):
    df2 = df.values.tolist()
    neigh = NearestNeighbors(n_neighbors=len(df))
    neigh.fit(df2)
    return neigh.kneighbors(df2, return_distance=False)

def tournament(pop:population, front, tournamentSize = 3):
    idxPar = []

    while len(idxPar) < 2:
        competingParentIdx = []
        score = []
        for _ in range(tournamentSize):
            idx = random.randint(0,pop.getSize()-1)
            while idx in competingParentIdx:
                idx = random.randint(0,pop.getSize()-1)
            competingParentIdx.append(idx)
            for idxFront in range(len(front)):
                if idx in front[idxFront]:
                    score.append(front[idxFront])
        if competingParentIdx[score.index(min(score))] not in idxPar:
            idxPar.append(competingParentIdx[score.index(min(score))])

    return idxPar

def geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP):
    pop = population()
    for _ in range(TAM_POP):
        pop.addMember(indiv())

    gen_no = 0
    neighborMatrix = getNeighbors(df)
    while(gen_no < MAX_GEN):

        pop.updateAllFitness(neighborMatrix, Cols, df)

        fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
        fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]

        non_dominated_sorted_solution = fast_non_dominated_sort(fit01Values[:],fit02Values[:])
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(fit01Values[:],fit02Values[:],non_dominated_sorted_solution[i][:]))

        # Create offsprint population
        popOffspring = population()
        while(popOffspring.getSize() != pop.getSize()):
            [idxParent01, idxParent02] = tournament(pop, non_dominated_sorted_solution)
            offsprint01, offsprint02 = pop.members[idxParent01].getCrossover(pop.members[idxParent02])
            offsprint01.mutate(neighborMatrix)
            offsprint02.mutate(neighborMatrix)
            popOffspring.addMember(offsprint01)
            popOffspring.addMember(offsprint02)

        popOffspring.updateAllFitness(neighborMatrix, Cols, df)
        
        popMix = population()
        popMix.mergePopulation(pop)
        popMix.mergePopulation(popOffspring)

        fit01ValuesMix = [popMix.members[i].fitness01 for i in range(popMix.getSize())]
        fit02ValuesMix = [popMix.members[i].fitness02 for i in range(popMix.getSize())]

        ## Ajustar... um dia...
        non_dominated_sorted_solutionMix = fast_non_dominated_sort(fit01ValuesMix[:],fit02ValuesMix[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solutionMix)):
            crowding_distance_values2.append(crowding_distance(fit01ValuesMix[:],fit02ValuesMix[:],non_dominated_sorted_solutionMix[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solutionMix)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solutionMix[i][j],non_dominated_sorted_solutionMix[i] ) for j in range(0,len(non_dominated_sorted_solutionMix[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solutionMix[i][front22[j]] for j in range(0,len(non_dominated_sorted_solutionMix[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==TAM_POP):
                    break
            if (len(new_solution) == TAM_POP):
                break
        
        pop = population()
        for idx in new_solution:
            pop.addMember(popMix.members[idx])
        
        gen_no += 1
    
    # prepara a fronteira final
    pop.updateAllFitness(neighborMatrix, Cols, df)
    fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
    fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]
    ## Ajustar... um dia...
    non_dominated_sorted_solution = fast_non_dominated_sort(fit01Values[:],fit02Values[:])
    pontos_function1 = []
    pontos_function2 = []
    for indiceIndiv in non_dominated_sorted_solution[0]:  # somente primeira fronteira
        pontos_function1.append(fit01Values[indiceIndiv])
        pontos_function2.append(fit02Values[indiceIndiv])
    return pontos_function1, pontos_function2

def preparaBD(arquivo):
    df = pd.read_csv(arquivo)
    Cols = len(df.columns)
    df2 = df.iloc[:,0:Cols-1]
    Cols = len(df2.columns)
    N = len(df2)
    return df2, N, Cols

def imprimeRodadas(matriz3D):
    plt.xlabel('f1', fontsize=15)
    plt.ylabel('f2', fontsize=15)
    xmin = matriz3D[0][0][0]
    xmax = matriz3D[0][0][0]
    ymin = matriz3D[0][1][0]
    ymax = matriz3D[0][1][0]
    for fronteira in matriz3D:
        pontos_function1 = fronteira[0]
        pontos_function2 = fronteira[1]
        # para encontrar limites do gráfico
        if min(pontos_function1) < xmin:
            xmin = min(pontos_function1)
        if max(pontos_function1) > xmax:
            xmax = max(pontos_function1)
        if min(pontos_function2) < ymin:
            ymin = min(pontos_function2)
        if max(pontos_function2) > ymax:
            ymax = max(pontos_function2)
        plt.scatter(pontos_function1, pontos_function2)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


if __name__ == "__main__":
    # run(GERACOES=200, TAM_POP=100, RODADAS=5)
    main(GERACOES=200, TAM_POP=100, RODADAS=5)
    # test()
