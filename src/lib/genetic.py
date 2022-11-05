# from src.genetic import population
from .readData import DataSource
from .representation import Population
from .operators import get_neighbor_matrix

def run_genetic(generations: int, ds: DataSource, initPopulation: Population=None):
    
    if not initPopulation:
        pop = Population()
    else:
        pop = initPopulation
    
    ds.neighborMatrix = get_neighbor_matrix(ds)

    for _ in range(generations):
        run_generation(pop, ds)


def run_generation(pop: Population, ds: DataSource) -> Population:
    pop.updateAllFitness(ds)





        

    #     fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
    #     fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]

    #     non_dominated_sorted_solution = nsgaII.fast_non_dominated_sort(fit01Values[:],fit02Values[:])
    #     crowding_distance_values=[]
    #     for i in range(0,len(non_dominated_sorted_solution)):
    #         crowding_distance_values.append(nsgaII.crowding_distance(fit01Values[:],fit02Values[:],non_dominated_sorted_solution[i][:]))

    #     # Create offsprint population
    #     popOffspring = population()
    #     while(popOffspring.getSize() != pop.getSize()):
    #         [idxParent01, idxParent02] = tournament(pop, non_dominated_sorted_solution)
    #         offsprint01, offsprint02 = crossover(pop.members[idxParent01],pop.members[idxParent02])
    #         offsprint01.mutate(neighborMatrix)
    #         offsprint02.mutate(neighborMatrix)
    #         popOffspring.addMember(offsprint01)
    #         popOffspring.addMember(offsprint02)

    #     popOffspring.updateAllFitness(neighborMatrix, Cols, dfValues)
        
    #     popMix = population()
    #     popMix.mergePopulation(pop)
    #     popMix.mergePopulation(popOffspring)

    #     fit01ValuesMix = [popMix.members[i].fitness01 for i in range(popMix.getSize())]
    #     fit02ValuesMix = [popMix.members[i].fitness02 for i in range(popMix.getSize())]

    #     ## Ajustar... um dia...
    #     non_dominated_sorted_solutionMix = nsgaII.fast_non_dominated_sort(fit01ValuesMix[:],fit02ValuesMix[:])
    #     crowding_distance_values2=[]
    #     for i in range(0,len(non_dominated_sorted_solutionMix)):
    #         crowding_distance_values2.append(nsgaII.crowding_distance(fit01ValuesMix[:],fit02ValuesMix[:],non_dominated_sorted_solutionMix[i][:]))
    #     new_solution= []
    #     for i in range(0,len(non_dominated_sorted_solutionMix)):
    #         non_dominated_sorted_solution2_1 = [nsgaII.index_of(non_dominated_sorted_solutionMix[i][j],non_dominated_sorted_solutionMix[i] ) for j in range(0,len(non_dominated_sorted_solutionMix[i]))]
    #         front22 = nsgaII.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
    #         front = [non_dominated_sorted_solutionMix[i][front22[j]] for j in range(0,len(non_dominated_sorted_solutionMix[i]))]
    #         front.reverse()
    #         for value in front:
    #             new_solution.append(value)
    #             if(len(new_solution)==TAM_POP):
    #                 break
    #         if (len(new_solution) == TAM_POP):
    #             break
        
    #     pop = population()
    #     for idx in new_solution:
    #         pop.addMember(popMix.members[idx])
    #     del popMix, popOffspring
    
    # # prepara a fronteira final
    # pop.updateAllFitness(neighborMatrix, Cols, dfValues)
    # fit01Values = [pop.members[i].fitness01 for i in range(pop.getSize())]
    # fit02Values = [pop.members[i].fitness02 for i in range(pop.getSize())]
    # ## Ajustar... um dia...
    # non_dominated_sorted_solution = nsgaII.fast_non_dominated_sort(fit01Values[:],fit02Values[:])
    # pontos_function1 = []
    # pontos_function2 = []
    # for indiceIndiv in non_dominated_sorted_solution[0]:  # somente primeira fronteira
    #     pontos_function1.append(fit01Values[indiceIndiv])
    #     pontos_function2.append(fit02Values[indiceIndiv])
    # return pontos_function1, pontos_function2, [pop.members[non_dominated_sorted_solution[0][i]].data for i in range(len(non_dominated_sorted_solution[0]))]
