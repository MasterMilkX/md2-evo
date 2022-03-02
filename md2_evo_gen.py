# Level generator for MiniDungeons2
# Generates levels using FI2Pop QD algorithm
# usage: python md2_evo_gen.py


# imports
import numpy as np
import random
import math
import sys
import subprocess
import os
from tqdm import tqdm
import json
from copy import deepcopy
from multiprocessing import Pool


#import fitness evaluator commands
from fitness_evaluator import evaluate_fitness


#global constants
USE_TEMPLATE = True       						#whether to use the preset template
EXT_GMA_CMD = "mono fitness_executable/MinidungeonsConsole.exe"     #external call to the program that evaluates the map and returns the result as a JSON
OUTPUT_FOLDER = "MD2_evo-archive"				#output folder for the exported levels created


# experiment parameters
ITERATIONS = 1000
MAP_MUTATION_RATE = 0.2
FEASIBLE_SEL_RATE = 0.6
POPSIZE = 1
MAX_SET_SIZE = POPSIZE
EMPTY_RATE = 0.5
WALL_RATE = 0.4
CORES = 5

PERSONA = "R"  # runner [R], monster_killer [MK], treasure_collector [TC]


# ascii char key
asc_char_map = {}
asc_char_map['empty'] = " "
asc_char_map['wall'] = "X"
asc_char_map['player'] = "H"
asc_char_map['blob'] = "b"
asc_char_map['minotaur'] = "m"
asc_char_map['exit'] = "e"
asc_char_map['trap'] = "t"
asc_char_map['treasure2'] = "T"
asc_char_map['potion'] = "P"
asc_char_map['portal'] = "p"
asc_char_map['ogre'] = "M"
asc_char_map['ogre2'] = "o"
asc_char_map['witch'] = "R"


# make choice characters (non-player and exit) and list of all characters
choice_chars = []
all_chars = []
for k, v in asc_char_map.items():
	if k not in ['player', 'exit']:
		choice_chars.append(v)
	all_chars.append(v)


# use this as the level template or import it from a file
template_map_str = "XXXXXXXXXX\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nX        X\nXXXXXXXXXX"
template_map = []
for r in template_map_str.split("\n"):
	y = [x for x in r]
	template_map.append(y)

# stores the minidungeons2 2d array ascii maps with fitness value and behavior characteristic


class EvoMap():
	def __init__(self, randomInit=False):
		# ascii map rep of the level (2d array of characters)
		self.asc_map = []
		self.fitness = 0  # fitness value
		self.con = 0  # constraint value
		# behavior characteristic definition [runner, monster killer, treasure collector]
		self.genome = PERSONA

		self.r = random.random()  # test key
		if randomInit:
			self.initRandomMap()
			# print(self)

	# makes a copy of the map

	def clone(self):
		new_map = EvoMap()
		new_map.asc_map = deepcopy(self.asc_map)
		new_map.fitness = self.fitness
		new_map.con = self.con
		new_map.r = self.r
		new_map.genome = self.genome

		return new_map

	# comparison function
	def __lt__(self, other):
		if self.con < other.con:
			return True
		elif self.con > other.con:
			return False
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
			f.write("%---%\n")
			f.write(f"fitness:{self.fitness}\n")
			f.write(f"con:{self.con}\n")
			f.write(f"genome:{self.genome}\n")

	# get a random position within these coordinates (x,y)
	def randPos(self, x1, x2, y1, y2):
		return (random.choice(range(x1, x2)), random.choice(range(y1, y2)))

	
	#makes sure there is only one character on the map and removes all others (randomly chosen)
	def singleConstraint(self, m, c):
		update_map = deepcopy(m)
		charPos = list(zip(*np.where(np.array(update_map) == asc_char_map[c])))  #list of tuples

		#too many characters - reduce them
		if len(charPos) > 1:
			sole_char = random.choice(charPos)

			for p in charPos:
				#skip the one char
				if p == sole_char:
					continue
				#change the character at the current position to something else
				update_map[p[0]][p[1]] = random.choice(choice_chars)

		#not enough of the character found
		elif len(charPos) == 0:	
			sole_char = self.randPos(1,len(m[0])-1,1,len(m)-1)   #just assume there are walls for ease
			update_map[sole_char[1]][sole_char[0]] = asc_char_map[c]



		#return the updated map with the single player
		return update_map

	#makes sure there is an even number of a characters on the map and removes the leftover (randomly chosen)
	def portalConstraint(self, m):
		update_map = deepcopy(m)
		charPos = list(zip(*np.where(np.array(update_map) == asc_char_map['portal'])))  #list of tuples

		sansportals = choice_chars.copy()
		sansportals.remove(asc_char_map['portal'])

		#too many characters - reduce them
		if len(charPos) > 2:
			sole_char = random.choices(charPos,k=2)

			for p in charPos:
				#skip the one char
				if p in sole_char:
					continue
				#change the character at the current position to something else (not portal)
				update_map[p[0]][p[1]] = random.choice(sansportals)

		#no pair - remove the single
		if len(charPos) == 1:
			update_map[charPos[0][0]][charPos[0][1]] = random.choice(sansportals)

		#return the updated map with 2 or 0 portals
		return update_map

	# creates a random map (using the template or something else)
	def initRandomMap(self, withWalls=True):
		am = []

		# use the empty template
		if USE_TEMPLATE:
			am = deepcopy(template_map)
			w = len(am[1])
			h = len(am)
		# create a whole new map with different dimensions
		else:
			w = random.randint(5, 20)
			h = random.randint(5, 20)
			for hi in range(h):
				row = []
				for wi in range(w):
					# make border of wall
					if (wi == 0 or wi == (w-1) or hi == 0 or hi == (h-1)):
						row.append(asc_char_map['wall'])
					# interior
					else:
						row.append(asc_char_map['empty'])
				am.append(row)

		# populate interior with random characters
		for hi in range(1, h-1):
			for wi in range(1, w-1):
				if(random.random() > EMPTY_RATE):
					if(random.random() > WALL_RATE):
						c = random.choice(choice_chars)
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
					nm[hi][wi] = random.choice(all_chars)

		# check for constraints
		nm = self.singleConstraint(nm, 'player')
		nm = self.singleConstraint(nm, 'exit')

		# make sure there are an even number of portals
		nm = self.portalConstraint(nm)

		#check character counts
		ch,cts = np.unique(np.array(nm),return_counts=True)
		for k,v in zip(ch,cts):
			print(f"{k} -> {v}")
		print("-----------")

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
			zip(*np.where(np.array(m) != asc_char_map['player'])))[0]
		reached = {}
		for e in emptyPos:
			#print(f"{e} -> {m[e[0]][e[1]]}")
			reached[e] = False

		s = heroStart  # start at a random empty tile position
		q = [s]  # initialize queue

		# flood fill with BFS
		while len(q) > 0:
			qi = q.pop()
			reached[qi] = True

			# get the neighbors and add to queue if valid
			n = self.getNeighbors(qi, m)
			for ni in n:
				if m[ni[0]][ni[1]] != asc_char_map['wall'] and reached[ni] == False:
					total_reached += 1
					q.append(ni)

		return total_reached / (1.0 * len(emptyPos))

	# check if the map can be traversed

	def canTraverse(self):
		m = self.asc_map

		# get all the empty tiles and set them as unreached
		emptyPos = list(
			zip(*np.where(np.array(m) != asc_char_map['wall'])))  # y,x

		reached = {}
		for e in emptyPos:
			#print(f"{e} -> {m[e[0]][e[1]]}")
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

	# evaluate the map for its constraint value, fitness value, and behavior characteristic
	def evalMap(self, fileID):
		# check the constraints first
		pc = self.getCharNum(asc_char_map['player']) == 1
		ec = self.getCharNum(asc_char_map['exit']) == 1
		porc = self.getCharNum(asc_char_map['portal']) % 2 == 0
		tc = self.ratioTraversible()

		# if pass all constraint checks, set to 1 otherwise 0
		if(pc and ec and porc and tc):
			self.con = 1
		else:
			self.con = tc

		# get the fitness value and behavior characteristic from the minidungeons simulator
		self.getExtEval(f"evomap-{fileID}.txt")  # file ID for parallelization

	# exports the ascii map to a text file specified
	def map2File(self, filename):
		with open(filename, 'w+') as f:
			s = ""
			for r in self.asc_map:
				s += f"{''.join(r)}\n"
			f.write(s.strip())

	# get the output from the external script call (to C# engine for MD2)
	def getExtEval(self, filename):
		self.map2File(filename)  # export the map out to be read by the engine
		os.popen(f"{EXT_GMA_CMD} {filename}")   #run the script

		#the script outputs a json file of the results - parse it
		filelabel = filename.replace('.txt','')
		jsonResFile = f"{filelabel}_results.txt"
		jsonObj = None
		with open(jsonResFile, 'r') as jf:
			jsonObj = json.load(jf)

		#pass json to the fitness function to get a value back
		fitness = evaluate_fitness(jsonObj,PERSONA)
		self.fitness = fitness


# QD algorithm for storing feasible-infeasible maps
class FI2Pop():
	def __init__(self, mapMutRate, maxSets, feasSelRate):
		self.feas = []  # feasible population (passes constraints)
		self.infeas = []  # infeasible population (does not pass constraints)
		self.population = []  # initialized evolved population
		self.mapMutate = mapMutRate  # mutation rate for the maps
		self.maxSets = maxSets  # maximum size of the feasible/infeasible population
		# probability to select from the feasible set for the new population
		self.feasRate = feasSelRate

	# initialize the population for the evolution
	def initPop(self, popsize):
		self.population = []
		for p in range(popsize):
			e = EvoMap(True)
			self.population.append(e)
		'''
		for p in self.population:
			print(f"{''.join(p.asc_map[1])}\n> r:{p.r:.2f}\n")
		'''

	# empty the archive
	def initArc(self):
		self.feas = []
		self.infeas = []

	# export the archive to a folder of text files with the information set
	def exportArc(self, folderLoc, expInfeas=False):
		# make the folder if it doesn't exist yet
		os.makedirs(folderLoc, exist_ok=True)

		# export the feasible population
		for i in range(self.feas):
			fm = self.feas[i]
			fm.exportMap(os.path.join(folderLoc, f"{PERSONA}-{i}-f.txt"))

		if expInfeas:
			# export the infeasible population
			for i in range(self.infeas[g]):
				fm = self.infeas[i]
				fm.exportMap(os.path.join(folderLoc, f"{PERSONA}-{i}-i.txt"))

	# sort the population into feasible or infeasible sets based on map info

	def sortPop(self):
		for p in self.population:
			# genome as a string binary value
			gs = "".join([str(pi) for pi in p.genome])
			c = p.con

			# TO FEASIBLE (IF CONSTRAINTS WORK AND BETTER FITNESS)
			if c == 1:
				self.feas.append(p.clone())

			# TO INFEASIBLE
			else:
				self.infeas.append(p.clone())

			# sort and trim off the worst constr/fit
			self.feas = sorted(self.feas, reverse=True)
			leftovers = deepcopy(self.feas[self.maxSets:])
			self.feas = self.feas[:self.maxSets]

			# add leftover feasibles to the infeasible population
			for li in leftovers:
				self.infeas.append(li)

			# trim infeasible set
			self.infeas = sorted(self.infeas, reverse=True)
			self.infeas = self.infeas[:self.maxSets]

	# create a new population by randomly selecting from the feasible or infeasible population
	def newPop(self, popsize, feasRate):
		newp = []
		for i in range(popsize):
			r = random.random()
			# select from the feasible population
			if r <= feasRate and len(self.feas) > 0:
				we = list(range(len(self.feas), 0, -1))
				newp.append(random.choices(self.feas, weights=we, k=1)[0])
			# select from the infeasible population
			else:
				we = list(range(len(self.infeas), 0, -1))
				newp.append(random.choices(self.infeas, weights=we, k=1)[0])

		# reset the population
		self.population = newp

	def evaluate_pop(self, parallel):
		if parallel:
			pop_map = [(i, p) for i, p in enumerate(self.population)]
			with Pool(CORES) as p:
				p.map(self.evaluate_chrome_helper, pop_map)
		else:
			i = 0
			for p in self.population:
				p.evalMap(i)
				i += 1

	def evaluate_chrome_helper(self, params):
		i = params[0]
		chromosome = params[1]
		chromosome.evalMap(i)

	# evolve the population and sort into feasible infeasible populations
	#mutate, evaluate, sort, repeat
	def evolvePop(self, popsize, iterations, parallel=False, outputArc=True):
		# reset the population and archive
		self.initPop(popsize)
		self.initArc()

		# for p in self.population:
		#	print(p)

		# start the iterations
		with tqdm(total=iterations) as pbar:
			for i in range(iterations):
				# take current population and mutate them
				for p in self.population:
					p.mutateMap(self.mapMutate)

				# evaluate the new population
				i = 0

				self.evaluate_pop(parallel)

				# sort into feasible and infeasible
				self.sortPop()

				# based on the iteration number - export to files
				if outputArc and (i+1) % 100 == 0:
					exportArc(OUTPUT_FOLDER)

				# select a new population
				self.newPop(popsize, self.feasRate)

				pbar.update(1)
				pbar.set_description(
					f"F: {[o.fitness for o in self.feas]} | Is: {[e.canTraverse() for e in self.infeas]}")


# run the main program if using this straight
if __name__ == "__main__":
	#print(f"$ {os.popen('python test.py Milk').read()}")

	expset = FI2Pop(MAP_MUTATION_RATE, MAX_SET_SIZE, FEASIBLE_SEL_RATE)

	print(" -------------   MD2 EVOLUTIONARY FI2POP EXPERIMENT   -------------")
	print("")
	print(
		f" ++                  PERSONA:  [ {PERSONA} ]                    ++")
	print("")
	print(f"> POPULATION SIZE:           {POPSIZE}")
	print(f"> ITERATIONS:                {ITERATIONS}")
	print(f"> OUTPUT_FOLDER:             {OUTPUT_FOLDER}")
	print(f"> MAP MUTATION RATE:         {MAP_MUTATION_RATE}")
	print(f"> EMPTY CHAR RATE:           {EMPTY_RATE}")
	print(f"> WALL CHAR RATE:            {WALL_RATE}")
	print(f"> MAX IN/FEASIBLE POP SIZE:  {MAX_SET_SIZE}")
	print(f"> FEASIBLE SELECTION RATE:   {FEASIBLE_SEL_RATE}")
	print("")

	expset.evolvePop(POPSIZE, ITERATIONS, parallel=True)
