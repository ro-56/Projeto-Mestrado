from math import floor
from random import choices
import numpy as np

from classes import individual
from operators import crossover
import population_manipulation as pop
import misc
import nsgaII
import graphs
from animation import plot_animation

def run(generation_size = 100, 
        population_size = 100,
        initial_population_ratios = {'kruscal': 1, 'kmedoid': 1, 'kmeans': 1},
        print_heatmap = False):

    # Get Dataset
    data_dataframe, answers_dataframe = misc.import_data('data/long1.csv')
    data_points = data_dataframe.values.tolist()
    answers_values = answers_dataframe.values.tolist()

    # Set parameters
    # generation_size = 100
    # population_size = 100
    # initial_population_ratios = {'kruscal': 1, 'kmedoid': 1, 'kmeans': 1}
    children_by_mutation = 30
    children_by_crossover = 30
    children_by_local_search = 10
    individual.neighbors = misc.get_neighbors(data_points)

    ##### TESTE #####
    fronteiras_nas_geracoes = []
    ######

    # Initialize population
    current_population, current_population_ratio = pop.initialize_population(population_size=population_size, population_ratio=initial_population_ratios, data_points=data_points)
    
    pop.update_fitness(current_population, data_points)

    # Run generations
    for g in range(generation_size):
        # Run generation
        current_population = run_generation(current_population, children_by_crossover, children_by_mutation, data_points, population_size)
        ######
        if g % 3 == 0:
            fitness_values = [indiv.fitness for indiv in current_population]
            indices = np.arange(len(current_population))
            indices_fronteira_nd = nsgaII.encontra_fronteira(fitness_values, indices)
            fronteiras_nas_geracoes.append([[i, fitness_values[i][0], fitness_values[i][1]] for i in indices_fronteira_nd])
        ######
    #  Analyse front
    fitness_values = [indiv.fitness for indiv in current_population]
    indices = np.arange(len(current_population))
    indices_fronteira_nd = nsgaII.encontra_fronteira(fitness_values, indices)
    populacao_nd = set([list(current_population)[i] for i in indices_fronteira_nd])
    # populacao_nd = current_population[indices_fronteira_nd, :]
    # qtde_nd = len(populacao_nd)
    fitness_nd = [[i] + fitness_values[i] for i in indices_fronteira_nd]

    fronts = graphs.separa_tres_regioes(fitness_nd)
    for front in fronts:
        front_idx = [item[0] for item in front]
        if print_heatmap:
            graphs.heatmap([list(populacao_nd)[i] for i in front_idx], answers_values)

    #####
    # plot_animation(fronteiras_nas_geracoes)
    #####
    return fitness_nd
    

def run_generation(current_population: set[individual], children_by_crossover: int, children_by_mutation: int, data_points, population_size: int) -> set[individual]:
    # Create children
    crossover_children = get_crossover_children(current_population, children_by_crossover)
    pop.update_fitness(crossover_children, data_points)

    mutation_children = get_mutation_children(current_population, children_by_mutation)
    pop.update_fitness(mutation_children, data_points)

    mixed_population = set()
    mixed_population.update(current_population)
    mixed_population.update(crossover_children)
    mixed_population.update(mutation_children)

    fitness_values = [indiv.fitness for indiv in mixed_population]
    new_population = set(nsgaII.selecao(list(mixed_population), fitness_values, population_size))

    return new_population


def get_mutation_children(current_population: set[individual], number_of_children: int) -> set[individual]:
    """
    Cria filhos a partir da mutação de indivíduos aleatórios da população
    """
    children_population = set()
    for _ in range(number_of_children):
        original_genotype = choices(list(current_population), k=1)[0].genotype
        random_index = np.random.randint(0, len(original_genotype))
        mutated_genotype = []
        for idx, org_val in enumerate(original_genotype):
            if idx == random_index:
                mutated_genotype.append(np.random.randint(1, max(original_genotype)+1))
            else:
                mutated_genotype.append(org_val)
        children_population.add(individual(genotype=mutated_genotype))
    return children_population


def get_crossover_children(current_population: set[individual], number_of_children: int) -> set[individual]:
    """
    Cria filhos a partir do cruzamento de indivíduos aleatórios da população
    """
    children_population = set()
    for _ in range(floor(number_of_children/2)):
        indiv1, indiv2 = choices(list(current_population), k=2)
        children = crossover(indiv1, indiv2)
        children_population.update(children)
    return children_population


if __name__ == "__main__":
    run()