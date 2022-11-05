from cProfile import label
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

import genetic
import misc
from classes import individual

data_dataframe, answers_dataframe = misc.import_data('data/long1.csv')
data_points = data_dataframe.values.tolist()
data_points_np = np.array(data_points)
answers_values = answers_dataframe.values.tolist()
 
gen_res = genetic.run(
    generation_size = 500, 
    population_size = 100,
    initial_population_ratios = {'kruscal': 1, 'kmedoid': 1, 'kmeans': 1})


spectral_indv = []
for i in range(2,8):
    clustering_spectral = SpectralClustering(n_clusters=i, assign_labels='discretize', random_state=1).fit(data_points_np)
    new_indiv = individual(genotype=clustering_spectral.labels_)
    new_indiv.calculate_fitness(data_points)
    spectral_indv.append(new_indiv)


# X_gaussian = np.array(data_points)
# clustering_gaussian = GaussianMixture(n_components=2, random_state=0).fit(X_gaussian)
# print(clustering_gaussian.)


kmeans_indv = []
for i in range(2,8):
    clustering_kmeans = KMeans(n_clusters=i, random_state=1).fit(data_points_np)
    new_indiv = individual(genotype=clustering_kmeans.labels_)
    new_indiv.calculate_fitness(data_points)
    kmeans_indv.append(new_indiv)


agglom_indv = []
for i in range(2,8):
    clustering_Agglom = AgglomerativeClustering(n_clusters=i).fit(data_points_np)
    new_indiv = individual(genotype=clustering_Agglom.labels_)
    new_indiv.calculate_fitness(data_points)
    agglom_indv.append(new_indiv)




_ = plt.subplots(figsize=(12,8))

f1 = [val[1] for val in gen_res]
f2 = [val[2] for val in gen_res]
plt.scatter(f1, f2, color='r', marker='+', label='Genetic Algorithm')


spectral_f1 = [indiv.fitness[0] for indiv in spectral_indv]
spectral_f2 = [indiv.fitness[1] for indiv in spectral_indv]
plt.scatter(spectral_f1, spectral_f2, color='b', marker='o', label='Spectral')


kmean_f1 = [indiv.fitness[0] for indiv in kmeans_indv]
kmean_f2 = [indiv.fitness[1] for indiv in kmeans_indv]
plt.scatter(kmean_f1, kmean_f2, color='g', marker='o', label='K-Means')


agglom_f1 = [indiv.fitness[0] for indiv in agglom_indv]
agglom_f2 = [indiv.fitness[1] for indiv in agglom_indv]
plt.scatter(agglom_f1, agglom_f2, color='k', marker='o', label='Agglomerative')

plt.legend(loc="upper right")
plt.show()