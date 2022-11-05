from lib.readData import read_data_file
from lib.representation import Individual, Population
from lib.genetic import run_genetic

def main():
    ds = read_data_file('data\\spiral.csv')
    
    Individual._maxNumCluster = 10
    Individual._numProxPoints = 150
    Individual._size = ds.number_atributes()

    Population._size = 10



    run_genetic(generations=1, ds=ds)

if __name__ == '__main__':
    main()