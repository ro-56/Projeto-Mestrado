from datetime import datetime
from math import floor
from pathlib import Path
import networkx as nx
import nsgaII
import matplotlib.pyplot as plt


from classes import indiv, population
from operators import crossover, tournament
from misc import getDistanceMatrix, getNeighbors, normalize_data, preparaBD, progressBar, solveKmeans, solveKmedoids, timer
from kruscal import make_population_with_kruscal_trees
# 
# ------------------


def desenhagrafo2d(lst, N, dfValues, idx):
    x = []
    y = []
    color = []

    for i in range(len(dfValues)):
        x.append(dfValues[i][0])
        y.append(dfValues[i][1])
        color.append(lst[i])
    plt.scatter(x, y, c=color)
    plt.savefig(tag_dir+'2d_'+str(idx)+'.png')
    plt.close()
    return

def printSolution(Xfront, Yfront, front) -> None:
    with open(f"{tag_dir}/solution.txt", 'w+') as f: 
        f.write(f"[[f1]]\n{','.join([str(i) for i in Xfront])}\n\n")
        f.write(f"[[f2]]\n{','.join([str(i) for i in Yfront])}\n\n")

        f.write(f"[[Front]]\n")
        for ind in front:
            f.write(f"{','.join([str(i) for i in ind])}\n")


def imprimeRodadas(matriz3D, title:str):
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
        plt.scatter(pontos_function1, pontos_function2, label=fronteira[2])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc="upper right")
    plt.title(title)
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
            color_map.append(res[node][0]+1)
            # if res[node][0]:
            #     color_map.append('red')
            # else:
            #     color_map.append('blue')
        else:
            color_map.append(0)
    
    fig = plt.figure(figsize=(50, 50))
    nx.draw(G, edge_color='black', with_labels=True, arrows=False,node_color=color_map) 
    # pylab.show()
    plt.savefig(tag_dir+'network_'+str(idx)+'.png')
    plt.close()
    return

def printHeatMap(res, fronts):

    def sortKey(list):
        return list[1]
    
    for idx, paretoFront in enumerate(fronts):
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
        plt.savefig(f"{tag_dir}heatmap{idx}.png")
        plt.close()

# -------------------
def geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP, kruscal_size=1, kmeans_size=1, kmedoids_size=1):

    dist = getDistanceMatrix(df.values.tolist())

    pop = initialize_population(df, TAM_POP, kruscal_size, kmeans_size, kmedoids_size, dist)

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

def initialize_population(df, TAM_POP, kruscal_size, kmeans_size, kmedoids_size, dist):
    pop = population()

    pop_fraction = [kruscal_size, kmeans_size, kmedoids_size]
    pop_fraction = [floor((pop_fraction[i]/sum(pop_fraction)) * TAM_POP) for i in range(len(pop_fraction))]

    # Parcela Kruscal
    ind = make_population_with_kruscal_trees(dist, 2, pop_fraction[0])
    for i in ind:
        pop.addMember(i)

    # Parcela kmeans
    numClusters = 2
    while pop.getSize() < (pop_fraction[0] + pop_fraction[1]):
        pop.addMember(indiv(solveKmeans(df.values.tolist(), numClusters)))
        numClusters += 1
    
    # Parcela kmedoids
    numClusters = 2
    while pop.getSize() < TAM_POP:
        pop.addMember(indiv(solveKmedoids(df.values.tolist(), numClusters)))
        numClusters += 1
    return pop

# 
# --------------------





@timer
def run(pointsFile: str, MAX_GEN = 10, TAM_POP = 10,
        frac_kmeans_init=[1],frac_kmedoids_init=[1], frac_kruscal_init=[1]):
    
    if len(frac_kmeans_init) != len(frac_kmedoids_init) \
        or len(frac_kmedoids_init) != len(frac_kruscal_init):
        raise ValueError(f"frac_kmeans_init, frac_kmedoids_init and frac_kruscal_init must have the same size\n{len(frac_kmeans_init)}-{len(frac_kmedoids_init)}-{len(frac_kruscal_init)}")

    df, N, Cols, res = preparaBD(pointsFile) 

    indiv.size = N

    prtAsw = False
    drwGph = False

    matriz3D = []
    fronts = []
    rounds = [frac_kruscal_init, frac_kmeans_init, frac_kmedoids_init]
    rounds = [[rounds[j][i] for j in range(len(rounds))] for i in range(len(rounds[0]))]
    for current in rounds:
        frac = TAM_POP/sum(current)
        nome_rodada = f"{(frac*current[0]):.2f}%kruscal-{(frac*current[1]):.2f}%KMeans-{(frac*current[2]):.2f}%Kmedoid"
        pontos_function1, pontos_function2, paretoFront = geneticoMultiobjetivo2(df, MAX_GEN, Cols, TAM_POP, current[0], current[1], current[2])
        matriz3D.append([pontos_function1, pontos_function2, nome_rodada])
        fronts.append(paretoFront)
    
    if prtAsw:
        printSolution(pontos_function1,pontos_function2,paretoFront)

    imprimeRodadas(matriz3D, pointsFile)
    printHeatMap(res.values.tolist(), fronts)

    # if drwGph:
    #     imprimeRodadas(matriz3D)
    #     printHeatMap(res.values.tolist(), paretoFront)
    #     for idx in range(len(paretoFront)):
    #         # desenhaGrafos(paretoFront[idx], N, res.values.tolist(), idx)
    #         desenhagrafo2d(paretoFront[idx], N, df.values.tolist(), idx)
    return

def setupDir():
    global tag_dir

    now = datetime.now()
    tag = now.strftime("%Y%m%d_%H%M%S")
    tag_dir = 'data/out/'+tag+'/'
    Path(tag_dir).mkdir(parents=True, exist_ok=True)


def main(data: str, MAX_GEN=1,TAM_POP=10,
        frac_kmeans_init=[1],frac_kmedoids_init=[1], frac_kruscal_init=[1]):
    
    indiv.maxNumCluster = 4
    indiv.numProxPoints = 25
    indiv.crossoverCuts = 2
    
    run(data, MAX_GEN, TAM_POP,
        frac_kmeans_init, frac_kmedoids_init, frac_kruscal_init)
    print('-----done-----')
    return
# 
# --------------------
tag_dir = ''
setupDir()
if __name__ == "__main__":
    main(data="data/long1.csv")