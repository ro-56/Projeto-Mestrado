import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def plot_animation(fitness_values):


    ind = [val[0] for val in fitness_values[0]]
    f1 = [val[1] for val in fitness_values[0]]
    f2 = [val[2] for val in fitness_values[0]]

    minF1 = np.min(f1)
    iMinF1 = np.argmin(f1)
    minF2 = np.min(f2)
    iMinF2 = np.argmin(f2)

    fig, ax = plt.subplots(figsize=(12,8)) 

    # for i in range(len(ind)):
    #     sc = plt.scatter(f1[i], f2[i], color='b', marker='+')

    filenames = []

    for f in enumerate(fitness_values):
        for i in range(len(ind)):
            sc = plt.scatter(f1[i], f2[i], color='b', marker='+')
            
        filename = f'{i}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    images = [imageio.imread(file_path) for file_path in filenames]
    imageio.mimsave('mygif.gif', images)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
