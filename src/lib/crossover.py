PREFIX = 'crossover_'

def crossover(xType: str):
    xTypes = [type.removeprefix(PREFIX) for type in [key for key in globals().keys() if PREFIX in key and callable(globals()[key])]]
    if xType not in xTypes:
        raise ValueError("Invalid crossover type. Expected one of: %s" % xTypes)
    
    globals()[f"{PREFIX}{xType}"]()

    
def crossover_1():
    print('1')
    return

def crossover_2():
    print('2')
    return

def crossover_3():
    print('3')
    return

def crossover_4():
    print('4')
    return

crossover('2')