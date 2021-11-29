import numpy as np

from misc import get_centroid


class individual:

    base_fitness = [0.0, 0.0]
    neighbors = []

    def __init__(self, genotype: list[int], fitness: list[float] = None):
        if fitness is None:
            fitness = self.base_fitness[:]

        self.genotype = genotype
        self.fitness = fitness

    def reset_fitness(self) -> None:
        """Limpa a fitness deste objeto"""
        self.fitness = self.base_fitness[:]
        return None


    def calculate_fitness(self, data_points):
        """Calcula a fitness deste objeto"""
        self.fitness[0] = self.__calculate_fitness_compactness(data_points)
        self.fitness[1] = self.__calculate_fitness_conectivity()
        return None

    def __calculate_fitness_compactness(self, data_points) -> float:
        """
        Calcula a fitness de um indivíduo
        """
        fitness = 0.0

        clusters = np.unique(self.genotype)
        for i, val in enumerate(clusters):
            cluster_points = [data_points[j] for j, point in enumerate(self.genotype) if point == val]
            centroids = get_centroid(cluster_points)
            fitness += np.linalg.norm(np.array(centroids) - np.array(cluster_points))

        return fitness


    def __calculate_fitness_conectivity(self, number_near_points: int = 10) -> float:
        """
        Calcula a fitness de um indivíduo
        """
        fitness = 0.0

        for i, _ in enumerate(self.genotype):
            for j in range(1, number_near_points+1):
                if self.genotype[i] != self.genotype[individual.neighbors[i][j]]:
                    fitness += 1/float(j+1)

        return fitness
