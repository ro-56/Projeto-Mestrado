import copy
import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import gc
import time
import cProfile
from pstats import Stats, SortKey
import time
from pathlib import Path

if __name__ == "__main__":
    import nsgaII
else:
    from . import nsgaII

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

# classes
# --------------------
class indiv:

    maxNumCluster = 0
    size = 0
    numProxPoints = 0
    crossoverCuts = 5

    def __init__(self, data:list[int] = []):
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
        self.members.append(indiv)

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

# Auxiliar Functions
# --------------------

def crossover(indiv_a:indiv, indiv_b:indiv):
        cuts = [random.randint(0,indiv_a.size-1) for _ in range(indiv_a.crossoverCuts)]
        a = [0 for _ in range(indiv_a.size)]
        b = [0 for _ in range(indiv_b.size)]
        invert = 0
        for i in range(indiv_a.size):
            if i in cuts:
                invert = not invert
            if not invert:
                a[i] = indiv_a.data[i]
                b[i] = indiv_b.data[i]
            else:
                a[i] = indiv_b.data[i]
                b[i] = indiv_a.data[i]

        return indiv(a), indiv(b)

def getAverageValue(vector):
    if not len(vector):
        return 0
    return sum(vector) / len(vector)

def kmeansNClusters(array, numClusters=2):
    return

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

# graphs
# --------------------
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
    # plt.show()
    plt.savefig(tag_dir+'paretoFront.png')
    plt.close()
    return

def desenhaGrafos(lst, N, res, idx=0):

    def isInt(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

    G = nx.Graph() 
    color_map = []
    
    for i in range(N):
        G.add_edges_from([(i, "C"+str(lst[i]))])

    for node in G:
        if isInt(node):
            if res[node][0]:
                color_map.append('red')
            else:
                color_map.append('blue')
        else:
            color_map.append('black')
    
    fig = plt.figure(figsize=(50, 50))
    nx.draw(G, edge_color='black', with_labels=True, arrows=False,node_color=color_map) 
    # pylab.show()
    plt.savefig(tag_dir+'network_'+str(idx)+'.png')
    plt.close()
    return

def printHeatMap(res, paretoFront):

    def sortKey(list):
        return list[1]

    Indvsize = len(paretoFront[0])
    size = len(paretoFront)
    mapMatrix = [[0 for _ in range(Indvsize)] for _ in range(Indvsize)]
    
    auxMap = [[i, res[i][0]] for i in range(Indvsize) ]
    auxMap.sort(key=sortKey,reverse=True)

    for i in range(size):
        for j in [auxMap[idx][0] for idx in range(Indvsize)]:
            for k in [auxMap[idx][0] for idx in range(Indvsize)]:
                if paretoFront[i][j] == paretoFront[i][k]:
                    mapMatrix[j][k] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(mapMatrix)
    # plt.show()
    plt.savefig(tag_dir+'heatmap.png')
    plt.close()

def normalize_data(lst):
    arr = np.asarray(lst)
    normArr = []
    for i in range(len(arr[0])):
        aux = []
        aux = (arr[:,i] - np.min(arr[:,i])) / (np.max(arr[:,i]) - np.min(arr[:,i]))
        normArr.append(aux)
    normArr = np.stack(normArr)
    return normArr.transpose().tolist()

# 
# ------------------
def geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP):
    pop = population([indiv() for _ in range(TAM_POP)])

    neighborMatrix = getNeighbors(df)
    dfValues = normalize_data(df.values.tolist())

    gens = list(range(MAX_GEN))
    for _ in progressBar(gens, prefix = 'Progress:', suffix = 'Complete'):

        pop.updateAllFitness(neighborMatrix, Cols, dfValues)

        fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
        fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]

        non_dominated_sorted_solution = nsgaII.fast_non_dominated_sort(fit01Values[:],fit02Values[:])
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(nsgaII.crowding_distance(fit01Values[:],fit02Values[:],non_dominated_sorted_solution[i][:]))

        # Create offsprint population
        popOffspring = population()
        while(popOffspring.getSize() != pop.getSize()):
            [idxParent01, idxParent02] = tournament(pop, non_dominated_sorted_solution)
            offsprint01, offsprint02 = crossover(pop.members[idxParent01],pop.members[idxParent02])
            offsprint01.mutate(neighborMatrix)
            offsprint02.mutate(neighborMatrix)
            popOffspring.addMember(offsprint01)
            popOffspring.addMember(offsprint02)

        popOffspring.updateAllFitness(neighborMatrix, Cols, dfValues)
        
        popMix = population()
        popMix.mergePopulation(pop)
        popMix.mergePopulation(popOffspring)

        fit01ValuesMix = [popMix.members[i].fitness01 for i in range(popMix.getSize())]
        fit02ValuesMix = [popMix.members[i].fitness02 for i in range(popMix.getSize())]

        ## Ajustar... um dia...
        non_dominated_sorted_solutionMix = nsgaII.fast_non_dominated_sort(fit01ValuesMix[:],fit02ValuesMix[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solutionMix)):
            crowding_distance_values2.append(nsgaII.crowding_distance(fit01ValuesMix[:],fit02ValuesMix[:],non_dominated_sorted_solutionMix[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solutionMix)):
            non_dominated_sorted_solution2_1 = [nsgaII.index_of(non_dominated_sorted_solutionMix[i][j],non_dominated_sorted_solutionMix[i] ) for j in range(0,len(non_dominated_sorted_solutionMix[i]))]
            front22 = nsgaII.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
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
        del popMix, popOffspring
    
    # prepara a fronteira final
    pop.updateAllFitness(neighborMatrix, Cols, dfValues)
    fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
    fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]
    ## Ajustar... um dia...
    non_dominated_sorted_solution = nsgaII.fast_non_dominated_sort(fit01Values[:],fit02Values[:])
    pontos_function1 = []
    pontos_function2 = []
    for indiceIndiv in non_dominated_sorted_solution[0]:  # somente primeira fronteira
        pontos_function1.append(fit01Values[indiceIndiv])
        pontos_function2.append(fit02Values[indiceIndiv])
    return pontos_function1, pontos_function2, [pop.members[non_dominated_sorted_solution[0][i]].data for i in range(len(non_dominated_sorted_solution[0]))]

# 
# --------------------
def setupProfiler():
    with cProfile.Profile() as pr:
            main(MAX_GEN=10,TAM_POP=50)

    with open(tag_dir+'/profiling_stats.txt', 'w') as stream:
        stats = Stats(pr, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats()
    return

def main(MAX_GEN=10,TAM_POP=10):
    run(MAX_GEN,TAM_POP)
    print('-----done-----')
    return

@timer
def run(MAX_GEN = 10, TAM_POP = 10, RODADAS = 1):
    df, N, Cols, res = preparaBD("data/diabetes.csv") 

    indiv.maxNumCluster = 4
    indiv.numProxPoints = 150
    indiv.size = N

    matriz3D = []
    for _ in range(RODADAS):
        pontos_function1, pontos_function2, paretoFront = geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP)
        matriz3D.append([pontos_function1, pontos_function2])
    imprimeRodadas(matriz3D)
    printHeatMap(res.values.tolist(), paretoFront)
    for idx in range(len(paretoFront)):
        desenhaGrafos(paretoFront[idx], N, res.values.tolist(), idx)
    return

def setupDir():
    global tag_dir

    now = datetime.now()
    tag = now.strftime("%Y%m%d_%H%M%S")
    tag_dir = 'data/out/'+tag+'/'
    Path(tag_dir).mkdir(parents=True, exist_ok=True)

# 
# --------------------
tag_dir = ''
setupDir()
if __name__ == "__main__":
    do_profiling = False
    if do_profiling:
        setupProfiler()
    else:
        main(MAX_GEN=100,TAM_POP=100)