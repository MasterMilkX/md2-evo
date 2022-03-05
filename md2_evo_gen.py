# Level generator for MiniDungeons2
# Generates levels using FI2Pop QD algorithm
# usage: python md2_evo_gen.py


# imports
import numpy as np
import random
import subprocess
import os
from tqdm import tqdm
import json
from copy import deepcopy
from multiprocessing import Pool
import yaml
import re
import csv

# import fitness evaluator commands
from fitness_evaluator import evaluate_fitness




# ascii char key
asc_char_map = {}
asc_char_map['empty'] = " "
asc_char_map['wall'] = "X"
asc_char_map['player'] = "H"
asc_char_map['blob'] = "b"
asc_char_map['minotaur'] = "m"
asc_char_map['exit'] = "e"
asc_char_map['trap'] = "t"
asc_char_map['treasure'] = "T"
asc_char_map['potion'] = "P"
asc_char_map['portal'] = "p"
asc_char_map['ogre'] = "M"
asc_char_map['ogre2'] = "o"
asc_char_map['witch'] = "R"


# makes initial random characters for map generation and mutation selection characters
INIT_ASCII_TILES = []
MUT_ASCII_TILES = []


# use this as the level template or import it from a file
template_map_str = "XXXXXXXXXX\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nXXXXXXXXXX"
template_map = []
for r in template_map_str.split("\n"):
    y = [x for x in r]
    template_map.append(y)
with open("template_map.txt") as f:
    template_map = []
    mapfile = f.read()
    for line in mapfile.split("\n"):
        row = [char for char in line]
        template_map.append(row)

# stores the minidungeons2 2d array ascii maps with fitness value and behavior characteristic


class EvoMap():
    def __init__(self, empty_rate, wall_rate, use_template, ext_cmd, persona, temp_chrome_folder, randomInit=False):
        # ascii map rep of the level (2d array of characters)
        self.asc_map = []
        self.fitness = 0  # fitness value
        self.con = 0  # constraint value
        # behavior characteristic definition [runner, monster killer, treasure collector]
        self.genome = persona
        self.ext_cmd = ext_cmd
        self.use_template = use_template
        self.empty_rate = empty_rate
        self.wall_rate = wall_rate
        self.temp_chrome_folder = temp_chrome_folder
        self.persona = persona
        self.r = random.random()  # test key
        if randomInit:
            self.initRandomMap()

    # makes a copy of the map

    def clone(self,saveStats=False):
        new_map = EvoMap(
            empty_rate=self.empty_rate,
            wall_rate=self.wall_rate,
            use_template=self.use_template,
            ext_cmd=self.ext_cmd,
            persona=self.persona,
            temp_chrome_folder=self.temp_chrome_folder,
            randomInit=False
        )
        new_map.asc_map = deepcopy(self.asc_map)
        if saveStats:
            new_map.fitness = self.fitness
            new_map.con = self.con
        else:
            new_map.fitness = 0
            new_map.con = 0
        new_map.genome = self.genome

        return new_map

    # comparison function
    def __lt__(self, other):
        if self.con != other.con:
            return self.con < other.con
        else:
            return self.fitness < other.fitness

    # to string built-in function
    def __str__(self):
        emstr = ""
        mstr = ""
        for r in self.asc_map:
            mstr += "".join(r) + "\n"
        emstr += (mstr)
        emstr += (f"c:{self.con}\n")
        emstr += (f"f:{self.fitness}\n")
        emstr += (f"g:{self.genome}\n")
        emstr += (f"r:{self.r}\n")

        return emstr

    # import a map from a file
    # assume map first with details separated by "%---%" line
    def importMap(self, filename):
        self.__init__()  # reset this object
        m = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            mf = True  # map flag
            for l in lines:
                # separator between details and ascii map
                if l == "%---%":
                    mf = False
                    self.asc_map = m
                    continue

                # add another row to the map
                if mf:
                    r = [x for x in l.strip()]
                    m.append(r)
                # set the details
                else:
                    if "fitness" in l:
                        self.fitness = float(l.split(":")[1])
                    elif "con" in l:
                        self.con = int(l.split(":")[1])
                    elif "genome" in l:
                        self.genome = l.split(":")[1]

    # export the map
    def exportMap(self, filename):
        # write the map first
        self.map2File(filename)

        # details second
        with open(filename, "a+") as f:
            f.write("\n%---%\n")
            f.write(f"fitness:{self.fitness}\n")
            f.write(f"con:{self.con}\n")
            f.write(f"genome:{self.genome}\n")

    # get a random position within these coordinates (x,y)
    def randPos(self, x1, x2, y1, y2):
        return (random.choice(range(x1, x2)), random.choice(range(y1, y2)))

    # makes sure there is only one character on the map and removes all others (randomly chosen)

    def singleConstraint(self, m, c):
        update_map = deepcopy(m)
        # list of tuples
        charPos = list(zip(*np.where(np.array(update_map) == asc_char_map[c])))

        # too many characters - reduce them
        if len(charPos) > 1:
            sole_char = random.choice(charPos)

            for p in charPos:
                # skip the one char
                if p == sole_char:
                    continue
                # change the character at the current position to something else
                update_map[p[0]][p[1]] = random.choice(MUT_ASCII_TILES)

        # not enough of the character found
        elif len(charPos) == 0:
            # just assume there are walls for ease
            sole_char = self.randPos(1, len(m[0])-1, 1, len(m)-1)
            update_map[sole_char[1]][sole_char[0]] = asc_char_map[c]

        # return the updated map with the single player
        return update_map

    # makes sure there is an even number of a characters on the map and removes the leftover (randomly chosen)
    def portalConstraint(self, m):
        update_map = deepcopy(m)
        # list of tuples
        charPos = list(zip(*np.where(np.array(update_map) == asc_char_map['portal'])))

        sansportals = MUT_ASCII_TILES.copy()
        if asc_char_map['portal'] in MUT_ASCII_TILES:
            sansportals.remove(asc_char_map['portal'])

        # too many characters - reduce them
        if len(charPos) > 2:
            random.shuffle(charPos)
            for p in charPos[2:]:
                # skip two
                # change the character at the current position to something else (not portal)
                update_map[p[0]][p[1]] = random.choice(sansportals)
            
        # no pair - remove the single
        if len(charPos) == 1:
            update_map[charPos[0][0]][charPos[0][1]] = random.choice(sansportals)

        # return the updated map with 2 or 0 portals
        charPos = list(zip(*np.where(np.array(update_map) == asc_char_map['portal'])))
        return update_map

    # creates a random map (using the template or something else)
    def initRandomMap(self, withWalls=True):
        am = []

        # use the empty template
        if self.use_template:
            am = deepcopy(template_map)
            width = len(am[1])
            height = len(am)
        # create a whole new map with different dimensions
        else:
            width = random.randint(5, 20)
            height = random.randint(5, 20)
            for hi in range(height):
                row = []
                for wi in range(width):
                    # make border of wall
                    if (wi == 0 or wi == (width-1) or hi == 0 or hi == (height-1)):
                        row.append(asc_char_map['wall'])
                    # interior
                    else:
                        row.append(asc_char_map['empty'])
                am.append(row)

        # populate interior with random characters
        for hi in range(1, height-1):
            for wi in range(1, width-1):
                if(random.random() > self.empty_rate):
                    if(random.random() > self.wall_rate):
                        c = random.choice(INIT_ASCII_TILES)
                    else:
                        c = asc_char_map['wall']

                else:
                    c = asc_char_map['empty']
                am[hi][wi] = c
        #########    PRESET CONSTRAINTS     ########

        # make sure there is only 1 player and 1 exit
        am = self.singleConstraint(am, 'player')
        am = self.singleConstraint(am, 'exit')

        # make sure there are an even number of portals
        am = self.portalConstraint(am)

        # set the ascii map to the generated map
        self.asc_map = am

    # mutates the ascii map according to a mutation rate
    def mutateMap(self, mutRate):
        nm = deepcopy(self.asc_map)
        for hi in range(1, len(nm)-1):
            for wi in range(1, len(nm[0])-1):
                if(random.random() <= mutRate):
                    nm[hi][wi] = random.choice(MUT_ASCII_TILES)

        # check for constraints
        nm = self.singleConstraint(nm, 'player')
        nm = self.singleConstraint(nm, 'exit')

        # make sure there are an even number of portals
        nm = self.portalConstraint(nm)

        # check character counts
        #ch, cts = np.unique(np.array(nm), return_counts=True)
        # for k, v in zip(ch, cts):
            # print(f"{k} -> {v}")
        # print("-----------")

        # replace with the new map
        self.asc_map = nm

    # get the quad-directional empty neighbors of a position

    def getNeighbors(self, p, m):
        # get quad-directions
        n = (p[0]-1, p[1])
        s = (p[0]+1, p[1])
        e = (p[0], p[1]+1)
        w = (p[0], p[1]-1)

        c = [n, s, e, w]
        neighbors = []

        # check if valid positions
        for ci in c:
            if ci[0] in range(len(m)) and ci[1] in range(len(m[0])):
                neighbors.append(ci)

        return neighbors

    # check total number of traversable tiles starting at a random tile
    def ratioTraversible(self):
        total_reached = 0
        m = self.asc_map

        # get all the empty tiles and set them as unreached
        emptyPos = list(
            zip(*np.where(np.array(m) != asc_char_map['wall'])))  # y,x

        heroStart = list(
            zip(*np.where(np.array(m) == asc_char_map['player'])))[0]
        reached = {}
        for e in emptyPos:
            # print(f"{e} -> {m[e[0]][e[1]]}")
            reached[e] = False

        s = heroStart  # start at a random empty tile position
        q = [s]  # initialize queue

        # flood fill with BFS
        while len(q) > 0:
            qi = q.pop()
            if not reached[qi]:
                reached[qi] = True
                total_reached += 1
            # get the neighbors and add to queue if valid
            n = self.getNeighbors(qi, m)
            for ni in n:
                if m[ni[0]][ni[1]] != asc_char_map['wall'] and reached[ni] == False:
                    q.append(ni)
        tc_result = total_reached / (1.0 * len(emptyPos))
        return tc_result

    # check if the map can be traversed

    def canTraverse(self):
        m = self.asc_map

        # get all the empty tiles and set them as unreached
        emptyPos = list(zip(*np.where(np.array(m) != asc_char_map['wall'])))  # y,x

        reached = {}
        for e in emptyPos:
            # print(f"{e} -> {m[e[0]][e[1]]}")
            reached[e] = False

        s = random.choice(emptyPos)  # start at a random empty tile position
        q = [s]  # initialize queue

        # flood fill with BFS
        while len(q) > 0:
            qi = q.pop()
            reached[qi] = True

            # get the neighbors and add to queue if valid
            n = self.getNeighbors(qi, m)
            for ni in n:
                if m[ni[0]][ni[1]] != asc_char_map['wall'] and reached[ni] == False:
                    q.append(ni)

        # check the reached
        for r in reached.values():
            if r == False:
                return False

        return True

    # get the exact number that a character appears in the map
    def getCharNum(self, c):
        return len(list(zip(*np.where(np.array(self.asc_map) == c))))

    def checkCon(self):
        # check the constraints first
        pc = self.getCharNum(asc_char_map['player']) == 1
        ec = self.getCharNum(asc_char_map['exit']) == 1
        porc = self.getCharNum(asc_char_map['portal']) % 2 == 0
        tc = 0
        if pc:
            # we need a hero for this to work
            tc = self.ratioTraversible()
        # if pass all constraint checks, set to 1 otherwise 0
        if(pc and ec and porc and tc==1):
            self.con = 1
        else:
            self.con = tc

    # evaluate the map for its constraint value, fitness value, and behavior characteristic
    def evalMap(self, fileID):
        # get the fitness value from the minidungeons simulator if valid constraints
        if self.con == 1:
            # file ID for parallelization
            return self.getExtEval(f"evomap-{fileID}.txt")
        else:
            #print(f"{fileID} invalid map!")
            #print(f"#{fileID}: {self.r} - {self.con:.2f} - {self.fitness:.2f}")
            return False

    # exports the ascii map to a text file specified
    def map2File(self, fileloc):
        with open(fileloc, 'w+') as f:
            mapstring = ""
            for r in self.asc_map:
                mapstring += f"{''.join(r)}\n"
            f.write(mapstring.strip())

    # get the output from the external script call (to C# engine for MD2)
    def getExtEval(self, filename):
        fileloc = os.path.join(self.temp_chrome_folder, filename)
        self.map2File(fileloc)  # export the map out to be read by the engine
        # run the script
        return subprocess.call(["mono", f"{self.ext_cmd}", f"{fileloc}"]) == 0   #make sure it works


# QD algorithm for storing feasible-infeasible maps
class FI2Pop():
    def __init__(self, config):
        self.feas = []  # feasible population (passes constraints)
        self.infeas = []  # infeasible population (does not pass constraints)
        self.population = []  # initialized evolved population
        self.mapMutate = config["MAP_MUTATION_RATE"]  # mutation rate for the maps
        self.maxSets = config["MAX_SET_SIZE"]  # maximum size of the feasible/infeasible population
        self.feasRate = config["FEASIBLE_SEL_RATE"] # probability to select from the feasible set for the new population    
        self.ext_cmd = config["EXT_GMA_CMD"]
        self.genome = config["PERSONA"]
        self.use_template = config["USE_TEMPLATE"]
        self.empty_rate = config["EMPTY_RATE"]
        self.wall_rate = config["WALL_RATE"]
        self.temp_chrome_folder = config["TEMP_CHROME_FOLDER"]
        self.persona = config["PERSONA"]
        self.cores = config["CORES"]
        self.func = config["FITNESS_FUNCTION"]
        self.output_folder = config["OUTPUT_FOLDER"]
        self.output_interval = config["OUTPUT_INTERVAL"]

    # initialize the population for the evolution
    def initPop(self, popsize):
        self.population = []
        for p in range(popsize):
            e = EvoMap(
                empty_rate=self.empty_rate,
                wall_rate=self.wall_rate,
                use_template=self.use_template,
                ext_cmd=self.ext_cmd,
                persona=self.persona,
                temp_chrome_folder=self.temp_chrome_folder,
                randomInit=True
            )
            self.population.append(e)


    # empty the archive
    def initArc(self):
        self.feas = []
        self.infeas = []

    # export the archive to a folder of text files with the information set
    def exportArc(self, folderLoc, label, expInfeas=True):
        # make the folder if it doesn't exist yet
        os.makedirs(folderLoc, exist_ok=True)

        # export the feasible population
        for i in range(len(self.feas)):
            fm = self.feas[i]
            fm.exportMap(os.path.join(folderLoc, f"{self.persona}-{i}-F_[{label}].txt"))

        if expInfeas:
            # export the infeasible population
            for i in range(len(self.infeas)):
                fm = self.infeas[i]
                fm.exportMap(os.path.join(folderLoc, f"{self.persona}-{i}-I_[{label}].txt"))

    # sort the population into feasible or infeasible sets based on map info

    def sortPop(self):
        for p in self.population:
            # genome as a string binary value
            gs = "".join([str(pi) for pi in p.genome])
            c = p.con

            # TO FEASIBLE (IF CONSTRAINTS WORK AND BETTER FITNESS)
            if c == 1:
                self.feas.append(p.clone(True))

            # TO INFEASIBLE
            else:
                self.infeas.append(p.clone(True))

            # sort and trim off the worst constr/fit
            self.feas = sorted(self.feas, reverse=True)
            leftovers = deepcopy(self.feas[self.maxSets:])
            self.feas = self.feas[:self.maxSets]

            # add leftover feasibles to the infeasible population
            for li in leftovers:
                self.infeas.append(li.clone(True))

            # trim infeasible set
            self.infeas = sorted(self.infeas, reverse=True)
            self.infeas = self.infeas[:self.maxSets]

    # create a new population by randomly selecting from the feasible or infeasible population
    def newPop(self, popsize, feasRate):
        newp = []
        for i in range(popsize):
            r = random.random()
            # select from the feasible population
            if (r <= feasRate and len(self.feas) > 0) or len(self.infeas) == 0:
                we = list(range(len(self.feas), 0, -1))
                newp.append(random.choices(self.feas, weights=we, k=1)[0].clone())
            # select from the infeasible population
            else:
                we = list(range(len(self.infeas), 0, -1))
                newp.append(random.choices(self.infeas, weights=we, k=1)[0].clone())

        # reset the population
        self.population = newp

    def evaluate_pop(self, parallel):
        #set the constraint value for the population
        for p in self.population:
            p.checkCon()

        if parallel:
            pop_map = [(i, p) for i, p in enumerate(self.population)]
            with Pool(self.cores) as p:
                p.map(self.evaluate_chrome_helper, pop_map)
        else:
            i = 0
            callres = {}
            for p in self.population:
                callres[i] = p.evalMap(i)
                i += 1
        
        # the script outputs a json file of the results - parse it
        for i, chromosome in enumerate(self.population):
            #failed, so skip
            try:
                filelabel = f"evomap-{i}"
                jsonResFile = f"{filelabel}_results.txt"
                jsonObj = None
                jsonResLoc = os.path.join(self.temp_chrome_folder, jsonResFile)
                with open(jsonResLoc, 'r') as jf:
                    jsonObj = json.load(jf)

                # pass json to the fitness function to get a value back
                fitness = evaluate_fitness(jsonObj, chromosome.asc_map, self.persona, func=self.func)
                chromosome.fitness = fitness
                #print(f"#{i}: {chromosome.r} - {chromosome.con:.2f} - {chromosome.fitness:.2f}")
            except Exception:
                continue


    def evaluate_chrome_helper(self, params):
        i = params[0]
        chromosome = params[1]
        chromosome.evalMap(i)

    # evolve the population and sort into feasible infeasible populations
    # mutate, evaluate, sort, repeat
    def evolvePop(self, popsize, iterations, parallel=False, outputArc=True):
        # reset the population and archive
        self.initPop(popsize)
        self.initArc()
        os.makedirs(self.temp_chrome_folder, exist_ok=True)

        #for tracking the fitness values over time
        best_fit = []
        avg_fit = []

        # start the iterations
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):

                #remove previous evomap files and results
                past_evofiles = [os.path.join(self.temp_chrome_folder,f) for f in os.listdir(self.temp_chrome_folder) if re.match('evomap-.+\.txt',f)]
                #print (f"( Deleting {len(past_evofiles)} previous evomap files... )")
                for pevo in past_evofiles:
                    os.remove(pevo)

                # take current population and mutate them
                for p in self.population:
                    p.mutateMap(self.mapMutate)

                # evaluate the new population
                self.evaluate_pop(parallel)
                for p in self.population:
                    print(p)

                #print(f"pop: {[f'{e.con}-{e.fitness}' for e in self.population]}")

                #grab the "elite" and average fitness from the population
                sortedPop = sorted(self.population,reverse=True)
                bestPopFit = sortedPop[0].fitness
                avgPopFit = np.average([e.fitness for e in self.population])

                best_fit.append(bestPopFit)
                avg_fit.append(avgPopFit)

                #confirm sorting
                #print([f"{e.con}-{e.fitness}" for e in self.population])
                #print(f"sorted pop: {[f'{e.con}-{e.fitness}' for e in sortedPop]}")

                # sort into feasible and infeasible
                self.sortPop()

                # based on the iteration number - export to files
                if outputArc and (i+1) % self.output_interval == 0:
                    self.exportArc(self.output_folder,f"Gen-{(i+1)}")

                # select a new population
                self.newPop(popsize, self.feasRate)

                pbar.update(1)
                pbar.set_description(f"Best: {bestPopFit} | Avg: {avgPopFit}")

        #export the best and averages as a csv in the archive folder
        with open(os.path.join(self.output_folder,'best_fit.csv'),'w+') as bf:
            c = csv.writer(bf)
            c.writerow(best_fit)

        with open(os.path.join(self.output_folder,'avg_fit.csv'),'w+') as af:
            c = csv.writer(af)
            c.writerow(avg_fit)



# run the main program if using this straight
if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    expset = FI2Pop(config)

    #set choice tiles for initializing map and mutating map
    INIT_ASCII_TILES = []
    MUT_ASCII_TILES = []
    for t in config['INIT_ASCII_TILES']:
        INIT_ASCII_TILES.append(asc_char_map[t])
    for t in config['MUT_ASCII_TILES']:
        MUT_ASCII_TILES.append(asc_char_map[t])


    print(" -------------   MD2 EVOLUTIONARY FI2POP EXPERIMENT   -------------")
    print("")
    print(
        f" ++                  PERSONA:  [ {config['PERSONA']} ]                    ++")
    print("")
    print(f"> ARCHIVE OUTPUT_FOLDER:     {config['OUTPUT_FOLDER']}")
    print(f"> TEMP EVO OUTPUT_FOLDER:    {config['TEMP_CHROME_FOLDER']}")
    print("")
    print(f"> MAP MUTATION RATE:         {config['MAP_MUTATION_RATE']}")
    print(f"> EMPTY CHAR RATE:           {config['EMPTY_RATE']}")
    print(f"> WALL CHAR RATE:            {config['WALL_RATE']}")
    print(f"> INIT ASCII TILES:          {config['INIT_ASCII_TILES']}")
    print(f"> MUT ASCII TILES:           {config['MUT_ASCII_TILES']}")
    print("")
    print(f"> MAX IN/FEASIBLE POP SIZE:  {config['MAX_SET_SIZE']}")
    print(f"> FEASIBLE SELECTION RATE:   {config['FEASIBLE_SEL_RATE']}")
    print("")
    print(f"> POPULATION SIZE:           {config['POPSIZE']}")
    print(f"> ITERATIONS:                {config['ITERATIONS']}")
    print(f"> PARALLEL:                  {config['PARALLEL']}")
    if config['PARALLEL']:
        print(f"> CORES:                     {config['CORES']}")
    print("")

    expset.evolvePop(config["POPSIZE"], config["ITERATIONS"], parallel=config['PARALLEL'], outputArc=config['OUTPUT_ARC'])
