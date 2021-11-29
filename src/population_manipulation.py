from math import floor

from classes import individual
import initialization_methods as init_methods
from kruscal import make_population_with_kruscal_trees


def initialize_population(population_size: int, population_ratio: dict, data_points) -> set[individual]:
    """
    Inicializa uma população com indivíduos
    Razão de cada tipo de indivíduo é definida pelo parâmetro population_ratio
    """
    allowed_methods = {'kruscal', 'kmedoid', 'kmeans'}
    unknown_methods = allowed_methods.union(population_ratio.keys()) - allowed_methods
    if unknown_methods:
        raise Exception('Population init method not implemented yet')

    population = set()
    number_of_each_type = {}

    for key in population_ratio:
        number_of_each_type[key] = floor((population_ratio.get(key)/sum(population_ratio.values())) * population_size)

    for key in population_ratio:
        if key == 'kruscal':
            # Parcela Kruscal
            method_population = set()
            method_population.update(make_population_with_kruscal_trees(data_points, 2, number_of_each_type.get(key)+1))

        elif key == 'kmedoid':
            # Parcela kmeans
            method_population = set()
            for i in range(number_of_each_type.get(key)):
                method_population.add(init_methods.make_kmedoids_individual(data_points, i+2))

        elif key == 'kmeans':
            # Parcela kmedoids
            method_population = set()
            for i in range(number_of_each_type.get(key)):
                method_population.add(init_methods.make_kmeans_individual(data_points, i+2))
    
        population.update(method_population)

    return population, number_of_each_type


def update_fitness(population: set, data_points) -> None:
    for element in population:
        element.calculate_fitness(data_points)
    return None