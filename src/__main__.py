import genetic

if __name__ == '__main__':
    genetic.main(data="data/iris.csv", TAM_POP=100, MAX_GEN=100,
                frac_kmeans_init=[1,2,1],frac_kmedoids_init=[1,1,1], frac_kruscal_init=[1,1,0])

#Cluters 2 -10
#TAm pop 100 100 200
#gen 100 400 400
