from random import random
from .representation import Individual
from .readData import DataSource
from sklearn.neighbors import NearestNeighbors

def get_neighbor_matrix(ds: DataSource):
    neigh = NearestNeighbors(n_neighbors=ds.number_data_points())
    neigh.fit(ds.data)
    return neigh.kneighbors(ds.data, return_distance=False)

def crossover(indvA: Individual, indvB: Individual) -> tuple[Individual, Individual]:
    pass

def mutate(indv: Individual, ds: DataSource) -> Individual:    
    
    def mutation01():
        return

    def mutation02():
        randMutationPoint = random.randint(0,indv._size-1)
        for idx in range(indv._size):
            if indv.genotype[randMutationPoint] != indv.genotype[ds.neighborMatrix[randMutationPoint][idx]]:
                indv.genotype[randMutationPoint] = indv.genotype[ds.neighborMatrix[randMutationPoint][idx]]
                break
    
    if (random.random() < 0.5):
        mutation01()
    else:
        mutation02()
    indv.resetFitness()



