import math
import numpy as np
import sys
import random

#read in the ascii file as a map
def readAsciiFile(filename):
    m = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.replace('\n','')
            m.append([x for x in l])
    return np.array(m)



#get the neighbors of a specific tile index
def getTileNeighbors(i,m):
    c = i % m.shape[1]
    r = math.floor(i / m.shape[1])
    
    n = []
    if r > 0:
        n.append(m[r-1][c])
    if r < (m.shape[0]-1):
        n.append(m[r+1][c])
    if c > 0:
        n.append(m[r][c-1])
    if c < (m.shape[1]-1):
        n.append(m[r][c+1])
        
    return n


#entropy fitness functions
def entr_fitness(m):
    num_tiles = np.prod(m.shape)

    # get tile entropy #
    tileCts = {}

    for r in m:
        for t in r:
            if t not in tileCts:
                tileCts[t] = 0
            tileCts[t]+=1

    tileProbs = {}
    for k,v in tileCts.items():
        tileProbs[k] = v/num_tiles

    entropy = sum([-1*(p * math.log10(p)) for p in tileProbs.values() if p > 0])

    #get tile derivative entropy (count # differing surrounding tiles)
    i = 0
    tileDiffCts = {}
    for a in range(5):  #4 directions, five possibilities
        tileDiffCts[a]=0

    for r in m:
        for t in r:
            n = getTileNeighbors(i,m)
            c = sum(map(lambda x: 0 if x == t else 1,n))
            tileDiffCts[c]+=1
            i+=1

    tileDiffProbs = {}
    for k,v in tileDiffCts.items():
        tileDiffProbs[k] = v/num_tiles

    der_entropy = -1*sum([(p * math.log10(p)) for p in tileDiffProbs.values() if p > 0])
    
    #return portion of both
    return 0.5*(1.0-entropy) #+0.5*(1.0-der_entropy)
    

#run program (like it was being called to the C# engine)
# persona_type = sys.argv[1]
# fileLoc = sys.argv[2]
# input_map = readAsciiFile(fileLoc)
# f = entr_fitness(input_map)
# fp = f*(random.randint(1,3)/3.0)   #simulate different fitnesses for the personas?

# #output back to the console(?)
# print(f"{fp}")






