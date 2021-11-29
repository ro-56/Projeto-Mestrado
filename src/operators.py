from classes import individual
from random import randint, choices

def crossover(parent_a: individual, parent_b: individual, number_of_cuts: int = 1) -> list[individual]:
    """
    Dois indivíduos são aleatóriamento selecionados da população
    São gerados dois filhos a partir do cruzamento dos indivíduos selecionados
    """

    chm_size = len(parent_a.genotype)
    child_a_chm = []
    child_b_chm = []

    # Seleciona aleatoriamente n ponto de corte
    cuts = choices([i for i in range(chm_size)], k=number_of_cuts)

    a_to_a = True
    for i in range(chm_size):
        if i in cuts:
            a_to_a = not a_to_a
        if a_to_a:
            child_a_chm.append(parent_a.genotype[i])
            child_b_chm.append(parent_b.genotype[i])
        elif not a_to_a:
            child_a_chm.append(parent_a.genotype[i])
            child_b_chm.append(parent_b.genotype[i])

    return [individual(child_a_chm), individual(child_b_chm)]
