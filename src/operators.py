from random import randint
from classes import indiv, population

def crossover(indiv_a:indiv, indiv_b:indiv):
    """"""
    cuts = [randint(0,indiv_a.size-1) for _ in range(indiv_a.crossoverCuts)]
    a = [0 for _ in range(indiv_a.size)]
    b = [0 for _ in range(indiv_b.size)]
    invert = 0
    for i in range(indiv_a.size):
        if i in cuts:
            invert = not invert
        if not invert:
            a[i] = indiv_a.data[i]
            b[i] = indiv_b.data[i]
        else:
            a[i] = indiv_b.data[i]
            b[i] = indiv_a.data[i]

    return indiv(a), indiv(b)

def tournament(pop:population, front, tournamentSize = 3):
    """"""
    idxPar = []

    while len(idxPar) < 2:
        competingParentIdx = []
        score = []
        for _ in range(tournamentSize):
            idx = randint(0,pop.getSize()-1)
            while idx in competingParentIdx:
                idx = randint(0,pop.getSize()-1)
            competingParentIdx.append(idx)
            for idxFront in range(len(front)):
                if idx in front[idxFront]:
                    score.append(front[idxFront])
        if competingParentIdx[score.index(min(score))] not in idxPar:
            idxPar.append(competingParentIdx[score.index(min(score))])

    return idxPar