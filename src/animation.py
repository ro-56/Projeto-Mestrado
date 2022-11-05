import matplotlib.pyplot as plt
import os
import imageio

def plot_animation(fitness_values):

    def norm(values):
        m = max(values)
        return [val/m for val in values]

    filenames = []
    last_x = []
    last_y = []
    first_x = []
    first_y = []
    for f, values in enumerate(fitness_values):
        _ = plt.subplots(figsize=(12,8))
        f1 = norm([val[1] for val in values])
        f2 = norm([val[2] for val in values])
        plt.scatter(f1, f2, color='b', marker='+')

        if f > 0:
            plt.scatter(last_x, last_y, color='#d3d3d3', marker='+')
            plt.scatter(first_x, first_y, color='#444444', marker='o')
        else:
            first_x = f1
            first_y = f2
        last_x = f1
        last_y = f2
            
        filename = f'{f}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    images = [imageio.imread(file_path) for file_path in filenames]
    imageio.mimsave('mygif0.gif', images, fps=4, loop=2)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
