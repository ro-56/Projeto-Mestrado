import numpy as np
import matplotlib.pyplot as plt

def separa_tres_regioes(fitness_values, print_graphs=0):
    ind = [val[0] for val in fitness_values]
    f1 = [val[1] for val in fitness_values]
    f2 = [val[2] for val in fitness_values]

    minF1 = np.min(f1)
    iMinF1 = np.argmin(f1)
    minF2 = np.min(f2)
    iMinF2 = np.argmin(f2)

    # Ponto 1
    Ponto1 = np.array([minF1, f2[iMinF1]])
    [a, b] = Ponto1
    
    # Ponto 2
    Ponto2 = np.array([f1[iMinF2], minF2])
    [c, d] = Ponto2
    
    # Ponto 3
    x1 = (c + 2*a)/3
    y1 = (d + 2*b)/3
    Ponto3 = np.array([x1, y1])
    
    # Ponto 4
    x2 = (2*c + a)/3
    y2 = (2*d + b)/3
    Ponto4 = np.array([x2, y2])
    
    # ============ Gráfico ==============
    if print_graphs:
        fig, ax = plt.subplots(figsize=(12,8)) 
        
        for i in range(len(ind)):
            plt.scatter(f1[i], f2[i], color='b', marker='+')
        plt.scatter(Ponto1[0], Ponto1[1], color='k', marker='o')
        plt.scatter(Ponto2[0], Ponto2[1], color='k', marker='o')
        plt.scatter(Ponto3[0], Ponto3[1], color='k', marker='o')
        plt.scatter(Ponto4[0], Ponto4[1], color='k', marker='o')
        plt.scatter(a, d, color='k', marker='o')
        
        # reta r que contém: P1, P2, P3, P4 (pontos equidistantes)
        x = np.linspace(a-1, c+1, 100)
        y = ((d-b)/(c-a))*(x-a) + b
        plt.plot(x, y, 'purple')
        
        # Reta1: passa pela solução utópica e por P1
        plt.plot([a]*len(y), y, 'purple')
        
        # Reta2: passa pela solução utópica e por P2
        plt.plot(x, [d]*len(x), 'purple')
        
        # Reta3: passa pela solução utópica e por P3
        x = np.linspace(a-1, x1+1, 100)
        yR3 = ((y1-d)/(x1-a))*(x-a) + d
        plt.plot(x, yR3, 'purple')
        
        # Reta4: passa pela solução utópica e por P4
        x = np.linspace(a-1, x2+1, 100)
        yR4 = ((y2-d)/(x2-a))*(x-a) + d
        plt.plot(x, yR4, 'purple')
        
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        plt.show()
    # ===================================
    
    # =========================================================================
    # Dividindo as Fronteiras: 1/3, 1/3 e 1/3
    fronteiraParte1 = [] 
    fronteiraParte2 = [] 
    fronteiraParte3 = []
    
    # para cada ponto da Fronteira completa
    for i in range(len(ind)):
        # Coloca na fronteira Parte1
        # se f1 está em [a,x1]
        # e f2 >= yR3
        if f1[i] <= x1 and f2[i] >= (((y1-d)/(x1-a))*(f1[i]-a) + d):
            fronteiraParte1.append([int(ind[i]), f1[i], f2[i]])
        # Senão, Coloca na fronteira Parte2
        # se f1 está em [a,x2]
        # e f2 >= yR4
        elif f1[i] <= x2 and f2[i] >= (((y2-d)/(x2-a))*(f1[i]-a) + d):
            fronteiraParte2.append([int(ind[i]), f1[i], f2[i]])
        # Senão, Coloca na fronteira Parte3
        else:
            fronteiraParte3.append([int(ind[i]), f1[i], f2[i]])
            
    # =========================================================================

    # ============ Gráfico ==============
    if print_graphs:
        fig2, ax2 = plt.subplots(figsize=(12,8)) 
        
        for i in range(len(fronteiraParte1)):
            plt.scatter(fronteiraParte1[i][1], fronteiraParte1[i][2], color='b', marker='+')
        
        for i in range(len(fronteiraParte2)):
            plt.scatter(fronteiraParte2[i][1], fronteiraParte2[i][2], color='r', marker='*')
            
        for i in range(len(fronteiraParte3)):
            plt.scatter(fronteiraParte3[i][1], fronteiraParte3[i][2], color='g', marker='4')
            
        plt.scatter(Ponto1[0], Ponto1[1], color='k', marker='o')
        plt.scatter(Ponto2[0], Ponto2[1], color='k', marker='o')
        plt.scatter(Ponto3[0], Ponto3[1], color='k', marker='o')
        plt.scatter(Ponto4[0], Ponto4[1], color='k', marker='o')
        plt.scatter(a, d, color='k', marker='o')
        
        # reta r que contém: P1, P2, P3, P4 (pontos equidistantes)
        x = np.linspace(a-1, c+1, 100)
        y = ((d-b)/(c-a))*(x-a) + b
        plt.plot(x, y, 'purple')
        
        # Reta1: passa pela solução utópica e por P1
        plt.plot([a]*len(y), y, 'purple')
        
        # Reta2: passa pela solução utópica e por P2
        plt.plot(x, [d]*len(x), 'purple')
        
        # Reta3: passa pela solução utópica e por P3
        x = np.linspace(a-1, x1+1, 100)
        yR3 = ((y1-d)/(x1-a))*(x-a) + d
        plt.plot(x, yR3, 'purple')
        
        # Reta4: passa pela solução utópica e por P4
        x = np.linspace(a-1, x2+1, 100)
        yR4 = ((y2-d)/(x2-a))*(x-a) + d
        plt.plot(x, yR4, 'purple')
        
        ax2.set_xlabel('f1')
        ax2.set_ylabel('f2')
        plt.show()
    
    return fronteiraParte1, fronteiraParte2, fronteiraParte3

def heatmap(population, data_points_answer) -> None:
    """broken: como determinar que o mesmo ponto em dois individuos estão em clusters iguais/diferentes se o 'nome' dos clusters podem ser diferetnes"""
    
  
    answers = [(i, data_points_answer[i]) for i, _ in enumerate(data_points_answer)]
    # answers.sort(key=sort_key)

    # x_label = y_label = [str(point[0]) for point in answers]

    z = np.zeros((len(answers), len(answers)))
    
    for indiv in population:
        for i, _ in answers:
            for j, _ in answers:
                if j > i:
                    if indiv.genotype[i] == indiv.genotype[j]:
                        z[i][j] += 1
                        z[j][i] = z[i][j]

    fig, ax = plt.subplots()
    
    map = ax.pcolormesh(z, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    # ax.set_xticks(np.arange(len(z))+0.5, minor=False)
    # ax.set_yticks(np.arange(len(z))+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # ax.set_xticklabels(x_label, minor=False)
    # ax.set_yticklabels(y_label, minor=False)
    plt.show()
    
    return None