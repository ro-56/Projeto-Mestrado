import numpy as np


def selecao(populacao: list, fitness: list, TP):
    fitness = np.array(fitness)
    populacao = np.array(populacao)

    # índices ainda não selecionados da populacoa mista
    indices_nao_selec = np.arange(len(populacao))  # qtde de individuos
    
    # todos os índices da populacao mista
    indices = np.arange(len(populacao))  
    
    # fronteira não-dominada
    fronteira_nd_indices = []
    
    # enquanto a fronteira for menor que TP
    while len(fronteira_nd_indices) < TP:  
        # encontra a nova fronteira, dentre os índices não selecionados
        nova_fronteira = encontra_fronteira(fitness[indices_nao_selec, :], indices_nao_selec)
        
        # total de soluções não-dominadas
        total_nd = len(fronteira_nd_indices) + len(nova_fronteira)

        # a última fronteira não coube toda
        if total_nd > TP:
            # quantidade que ainda pode ser inserida
            qtde = TP - len(fronteira_nd_indices)
            solucoes_selecionadas = __seleciona_com_crowding(fitness[nova_fronteira], qtde)
            nova_fronteira = nova_fronteira[solucoes_selecionadas]
        
        fronteira_nd_indices = np.hstack((fronteira_nd_indices, nova_fronteira))
        restantes = set(indices) - set(fronteira_nd_indices)
        indices_nao_selec = np.array(list(restantes))
        
    populacao_selecionada = populacao[fronteira_nd_indices.astype(int)]

    return populacao_selecionada


# encontrar o índice das soluções não-dominadas 
def encontra_fronteira(fitness, indices):
    if type(fitness) is not np.ndarray:
        fitness = np.array(fitness)
    if type(indices) is not np.ndarray:
        indices = np.array(indices)

    tam_pop = fitness.shape[0]
    fronteira_nd = np.ones(tam_pop, dtype=bool)    # todos True=1 inicialmente
    for i in range(tam_pop):  # solução i
        for j in range(tam_pop):   # solução j
            # Minimização!
            # j não é pior que i em nenhum dos objetivos e é melhor em pelo menos um
            if all(fitness[j] <= fitness[i]) and any(fitness[j] < fitness[i]):
                # i não faz parte da fronteira de Pareto porque j domina i
                fronteira_nd[i] = 0               # muda para False=0
                break

    return indices[fronteira_nd]


def __seleciona_com_crowding(fitness, qtde):
    # atribui índices às soluções sob análise: da última fronteira
    indices = np.arange(fitness.shape[0])
    
    # soluções que serão selecionadas
    solucoes_selecionadas = np.zeros(qtde)
    
    # calcula o crowding distance
    crowding_distance = __crowding(fitness)
    
    # do maior para o menor
    crowding_distance_ordenado = -np.sort(-crowding_distance)
    indices_ordenado = np.argsort(-crowding_distance)
    
    # para a quantidade que deve ser selecionada
    for i in range(qtde):
        solucoes_selecionadas[i] = indices_ordenado[i]
        
    solucoes_selecionadas = np.array(solucoes_selecionadas, dtype=int)

    return solucoes_selecionadas


def __crowding(fitness):
    tam_pop = len(fitness[:, 0])   # quantidade de pontos
    funcoes = len(fitness[0, :])   # quantidade de funções-objetivo               
    matriz_crowding = np.zeros((tam_pop, funcoes))  
    # fitness.ptp(0) array com máximos de cada coluna
    fitness_normalizado = (fitness - fitness.min(0))/fitness.ptp(0)  
    
    # para cada função-objetivo
    for i in range(funcoes):
        crowding_resultado = np.zeros(tam_pop)
        
        # pontos extremos tem o maior resultado de crowding
        crowding_resultado[0] = 1             
        crowding_resultado[tam_pop - 1] = 1    
        
        fitness_normalizado_ordenado = np.sort(fitness_normalizado[:,i])
        indices_fitness_normalizado = np.argsort(fitness_normalizado[:,i])
        
        # crowding distance: solução i, crowding = fitness[i+1] - fitness[i-1]
        crowding_resultado[1:tam_pop-1] = (fitness_normalizado_ordenado[2:tam_pop] - fitness_normalizado_ordenado[0:tam_pop-2])
        reordenar = np.argsort(indices_fitness_normalizado)
        matriz_crowding[:, i] = crowding_resultado[reordenar]
    
    crowding_distance = np.sum(matriz_crowding, axis=1) 

    return crowding_distance