##%%
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


def timer(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        ret = func(*args, **kwargs)
        delta = time.time() - startTime
        print(func.__name__, "elapsed time: ", delta)
        return ret
    return wrapper

##%%
def geneticoMultiobjetivo(populacao, TAM_POP, GERACOES, df, Cols):
    gen_no = 0   # geração número 0
    neigh = getNeighbors(df)
    while(gen_no < GERACOES):
        # function1_values é uma lista com todos os valores de f1 de todos os indivíduos
        function1_values = [function1(populacao[i], df, Cols) for i in range(TAM_POP)]
        # function2_values é uma lista com todos os valores de f2  de todos os indivíduos
        function2_values = [function2(populacao[i], df, Cols, neigh) for i in range(TAM_POP)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
        # print("Indivíduos na primeira Fronteira na geração",gen_no,":", len(non_dominated_sorted_solution[0]))
        # print("Índices dos indivíduos da primeira Fronteira: (ind, f1, f2)")
        # for indiceIndiv in non_dominated_sorted_solution[0]:
        #     print((indiceIndiv, function1_values[indiceIndiv], function2_values[indiceIndiv])) 
        # print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        # print("População dividida em Fronteiras: ")
        # print(non_dominated_sorted_solution)
        # print("Crowding Distance da População (por fronteira): ")
        # print(crowding_distance_values)
        # montando a população mista (solution2): população atual + filhos
        solution2 = populacao[:]  # copia população atual toda
        # gerando os filhos
        while(len(solution2)!=2*TAM_POP):
            pai1 = random.randint(0,TAM_POP-1)  # seleciona um índice
            pai2 = random.randint(0,TAM_POP-1)
            # coloca na mista os filhos gerados no crossover e mutação
            filho1, filho2 = crossover(populacao[pai1], populacao[pai2])
            filho1 = mutation(filho1,neigh)
            filho2 = mutation(filho2,neigh)
            solution2.append(filho1)
            solution2.append(filho2)
        # avalia a população mista toda
        function1_values2 = [function1(solution2[i], df, Cols) for i in range(0,2*TAM_POP)]
        function2_values2 = [function2(solution2[i], df, Cols, neigh) for i in range(0,2*TAM_POP)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==TAM_POP):
                    break
            if (len(new_solution) == TAM_POP):
                break
        populacao = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1
    # prepara a fronteira final
    # function1_values é uma lista com todos os valores de f1 de todos os indivíduos
    function1_values = [function1(populacao[i], df, Cols) for i in range(TAM_POP)]
    # function2_values é uma lista com todos os valores de f2 de todos os indivíduos
    function2_values = [function2(populacao[i], df, Cols, neigh) for i in range(TAM_POP)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    pontos_function1 = []
    pontos_function2 = []
    for indiceIndiv in non_dominated_sorted_solution[0]:  # somente primeira fronteira
        pontos_function1.append(function1_values[indiceIndiv])
        pontos_function2.append(function2_values[indiceIndiv])
    return pontos_function1, pontos_function2

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

# @timer
def getNeighbors(df):
    df2 = df.values.tolist()
    neigh = NearestNeighbors(n_neighbors=len(df))
    neigh.fit(df2)
    return neigh.kneighbors(df2, return_distance=False)

# @timer
def function2(x, df, Cols, neigh):
    L = 10
    df2 = df.values.tolist()
    totalSum = 0
    
    for i in range(len(x)):
        for l in range(1,L+1):
            if x[neigh[i][l]] == x[i]:
                continue
            totalSum += 1/float(l+1)
    return totalSum

# @timer   
def function1(x, df, Cols):
    points = df.values
    NClusters = max(x)
    pointInCluster = [[] for _ in range(NClusters)]
    clusterCenters = []
    for cl in range(NClusters):
        center = np.zeros(Cols)
        counter = 0
        for i in range(len(x)):
            if x[i] != cl:
                continue
            counter = counter + 1
            center = points[i] + center
            pointInCluster[cl].append(points[i])
        clusterCenters.append(center/counter)
    
    finalsum = 0
    for cl in range(NClusters):
        if not pointInCluster[cl]:
            continue
        finalsum += np.linalg.norm(clusterCenters[cl] - pointInCluster[cl])
    return finalsum    

def inicializaIndividuoAleatorio(N, CLUSTERS=30):
    individuo = [0]*N
    for a in range(N):
        individuo[a] = random.randint(0,CLUSTERS)
    return individuo
    
def inicializaPopulacaoAleatoria(TAM_POP, N, CLUSTERS=30):
    populacao = []
    CLUSTERS = 15  # TESTAR DIFERENTES VALORES
    for i in range(TAM_POP):
        individuo = inicializaIndividuoAleatorio(N, CLUSTERS)
        # individuo = inicializaIndividuoAleatorio(CLUSTERS)
        populacao.append(individuo)
    return populacao

def preparaBD(arquivo):
    df = pd.read_csv(arquivo)
    Cols = len(df.columns)
    df2 = df.iloc[:,0:Cols-1]
    Cols = len(df2.columns)
    N = len(df2)
    return df2, N, Cols

def mutation(indv, neigh):
    if (random.random() < 0.5):
        a = mutation1(indv)
    else:
        a = mutation2(indv, neigh)
    return a

def mutation1(indv):
    # N = len(indv)
    return indv

def mutation2(indv, neigh):
    N = len(indv)
    randMutation1 = random.randint(0,N-1)
    for i in range(N):
        if indv[neigh[randMutation1][i]] != indv[randMutation1]:
            indv[randMutation1] = indv[neigh[randMutation1][i]]
            break
    return indv

def crossover(indv1, indv2):
    N = len(indv1)
    CUTS = 5
    cuts = [random.randint(0,N-1) for _ in range(CUTS)]
    a = [0 for _ in range(N)]
    b = [0 for _ in range(N)]
    invert = 0
    for i in range(N):
        if i in cuts:
            invert = not invert
        if not invert:
            a[i] = indv1[i]
            b[i] = indv2[i]
        else:
            a[i] = indv2[i]
            b[i] = indv1[i]

    return [a, b]

@timer
def run(GERACOES=1, TAM_POP=4, RODADAS=1):
    df, N, Cols = preparaBD("diabetes.csv")  
    matriz3D = []
    for r in range(RODADAS):
        populacao = inicializaPopulacaoAleatoria(TAM_POP, N)
        pontos_function1, pontos_function2 = geneticoMultiobjetivo(populacao, TAM_POP, GERACOES, df, Cols)
        matriz3D.append([pontos_function1, pontos_function2])
    imprimeRodadas(matriz3D)

# função principal: chama as demais funções
def main(): 
    # run(GERACOES=1, TAM_POP=4, RODADAS=1)
    # # definir parâmetros do Genético
    GERACOES = 1
    TAM_POP = 4
    RODADAS = 1  
    # # leitura e preparação da base de dados: 
    df, N, Cols = preparaBD("diabetes.csv")  
    matriz3D = []  # matriz com todas as fronteiras
    for r in range(RODADAS):
    #     # Genético
        populacao = inicializaPopulacaoAleatoria(TAM_POP, N)
        # print(function1(populacao[0], df, Cols))
    #     # populacao = inicializaPopulacaoKMEANS(df, N, 0, TAM_POP)
    #     # populacao = populacao1 + populacao2
    #     # print("Individuo", populacao)
    #     # print("Individuo 1")        
    #     # desenhaClusters(populacao[0], N)
    #     # print("Individuo 2")
    #     # desenhaClusters(popula-cao[1], N)
    #     # print("Individuo 3")
    #     # desenhaClusters(populacao[2], N)
    #     # print("Individuo 4")
    #     # desenhaClusters(populacao[3], N)
        pontos_function1, pontos_function2 = geneticoMultiobjetivo(populacao, TAM_POP, GERACOES, df, Cols)
        matriz3D.append([pontos_function1, pontos_function2])
    imprimeRodadas(matriz3D)
    return


##%%
if __name__ == "__main__":
    run(GERACOES=200, TAM_POP=100, RODADAS=5)
    # main()


