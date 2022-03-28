# Level generator for MiniDungeons2
# Generates levels and categorizes using MapElites
# usage: python md2_mapelites_gen.py


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

#import OG chromosome of evovled map
from md2_evo_gen import EvoMap, FI2Pop, INIT_ASCII_TILES, MUT_ASCII_TILES


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


#runs the MAP-Elites algorithm using the feasible, infeasible population
class MAPElites():
	def __init__(self,config):
		# matrices made up of EvoMap objects with a vector key for their behavior characterstic
		self.feas_matrix = {}
		self.infeas_matrix = {}

		self.eliteProb = config['ELITE_PROB']  #probability to select from the elite (feas) dataset
		self.bc_dim_size = config['MAP_ELITES_DIM']  #dimensionality size of the matrix axis
		self.cell_size = config['MAX_CELL_SIZE']
		self.temp_chrome_folder = config["TEMP_CHROME_FOLDER"]
		self.output_folder = config['OUTPUT_FOLDER']
		self.config = config
		self.fi2 = FI2Pop(config)

	#divide the matrixes into elite and non-elite
	def divideMaps(self):
		elites = []
		feasibles = []
		infeasibles = []

		#sort to get the elites first
		for mc in self.feas_matrix.values():
			sort_mc = sorted(mc,reverse=True)
			elites.append(sort_mc[0])

			#add the others to feasibles
			for x in sort_mc[1:]:
				feasibles.append(x)

		#add all the infeasibles (sorting doesn't matter)
		for mc in self.infeas_matrix.values():
			for x in mc:
				infeasibles.append(x)

		return {'elites':elites, "feas": feasibles, "infeas": infeasibles}


	#convert a raw binary vector (arr) to a string format for the matrix key
	def vec2str(self,vec):
		return "".join([str(x) for x in vec])

	#get the feasible/infeasible population for a given characteristic vector
	def getCell(self,vec):
		vs = self.vec2str(vec)
		cell = {'feas':[],'infeas':[]}
		if vs in self.feas_matrix:
			cell['feas'] = self.feas_matrix.copy()
		if vs in self.infeas_matrix:
			cell['infeas'] = self.infeas_matrix.copy()
		return cell

	#convert the behavior characterstic value to a vector representation for the MAP-Elites
	def bcvalue2vec(self, dimensions):
		##    REPLACE ME!   ##
		z = []
		bins = [x for x in np.arange(self.bc_dim_size, 1, self.bc_dim_size)]
		for idx, characteristic in enumerate(dimensions):
			z.append(np.digitize(characteristic, bins=bins))
		return z


	def evaluate_pop(self, parallel):
		#set the constraint value for the population
		for p in self.fi2.population:
			p.checkCon()

		if parallel:
			pop_map = [(i, p) for i, p in enumerate(self.fi2.population)]
			with Pool(self.fi2.cores) as p:
				p.map(self.fi2.evaluate_chrome_helper, pop_map)
		else:
			i = 0
			callres = {}
			for p in self.fi2.population:
				callres[i] = p.evalMap(i)
				i += 1

		# the script outputs a json file of the results - parse it
		for i, chromosome in enumerate(self.fi2.population):
			try:
				filelabel = f"evomap-{i}"
				jsonResFile = f"{filelabel}_results.txt"
				jsonObj = None
				jsonResLoc = os.path.join(self.temp_chrome_folder, jsonResFile)
				with open(jsonResLoc, 'r') as jf:
					jsonObj = json.load(jf)

				# pass json to the fitness function to get a value back
				fitness, dimensions = evaluate_fitness(jsonObj, chromosome.asc_map, "None", "simplicity", config=config)
				#assign to the chromosome
				chromosome.fitness = fitness
				behavior_vector = self.bcvalue2vec(dimensions)
				chromosome.bvec = behavior_vector

			except Exception:
				continue


	#puts the population in the associated matrix and cells based on the behavior characterstic
	def pop2Matrix(self):

		#sort the feasible to get the elites (later)
		for p in self.fi2.feas:
			bv = p.bvec
			bv_str = self.vec2str(bv)

			#key not exists yet, add it
			if bv_str not in self.feas_matrix:
				self.feas_matrix[bv_str] = []

			#add to the set
			self.feas_matrix[bv_str].append(p.clone(True))

			#trim down
			self.feas_matrix[bv_str] = sorted(self.feas_matrix[bv_str],reverse = True)
			leftovers = []
			if len(self.feas_matrix[bv_str]) > self.cell_size:
				leftovers = deepcopy(self.feas_matrix[bv_str][self.cell_size:])
			self.feas_matrix[bv_str] = self.feas_matrix[bv_str][:self.cell_size]

			#add the leftovers to the infeasible set
			for l in leftovers:
				lbv = l.bvec
				lbv_str = self.vec2str(lbv)

				#key not exists yet, add it
				if lbv_str not in self.infeas_matrix:
					self.infeas_matrix[lbv_str] = []

				#add to the set
				self.infeas_matrix[lbv_str].append(l.clone(True))



		#sort the infeas to the matrix
		for p in self.fi2.infeas:
			bv = p.bvec
			bv_str = self.vec2str(bv)

			#key not exists yet, add it
			if bv_str not in self.infeas_matrix:
				self.infeas_matrix[bv_str] = []

			#add to the set
			self.infeas_matrix[bv_str].append(p.clone(True))

			#trim down
			self.infeas_matrix[bv_str] = sorted(self.infeas_matrix[bv_str],reverse = True)
			self.infeas_matrix[bv_str] = self.infeas_matrix[bv_str][:self.cell_size]


		#  ------ DEBUG -----  #
		#new_map_set = self.divideMaps()
		#print(f"# Elite:{len(new_map_set['elites'])} | # Feas:{len(new_map_set['feas'])} | # Infeas:{len(new_map_set['infeas'])}")
		#print(f"keys feas: {list(self.feas_matrix.keys())} | keys infeas: {list(self.infeas_matrix.keys())}")


	#generate a new population from the elites with a rate (retrieves from the other items otherwise)
	def newPop(self, popsize, eliteRate, feasRate):
		newp = []

		#get the separated sets
		map_sets = self.divideMaps()

		#create the new population
		for i in range(popsize):
			r = random.random()

			# select from the elite population
			if (r <= eliteRate and len(map_sets['elites']) > 0):
				newp.append(random.choice(map_sets['elites']).clone(True))
			# otherwise select from the rest of the population
			else:
				r2 = random.random()
				if (r2 <= feasRate and len(map_sets['feas']) > 0):
					newp.append(random.choice(map_sets['feas']).clone(True))
				else:
					newp.append(random.choice(map_sets['infeas']).clone(True))

		return newp



	#evolve a population of EvoMaps (similar to FI2Pop class) and place into matrices based on behavioral characteristic
	def evolvePop(self, popsize, iterations, parallel=False, outputArc=True):
		# reset the population and archive
		self.fi2.initPop(popsize)
		self.fi2.initArc()
		os.makedirs(self.fi2.temp_chrome_folder, exist_ok=True)

		#for tracking the fitness values over time
		best_fit = []
		avg_fit = []

		# start the iterations
		with tqdm(total=iterations) as pbar:
			for i in range(iterations):

				#remove previous evomap files and results
				past_evofiles = [os.path.join(self.fi2.temp_chrome_folder,f) for f in os.listdir(self.fi2.temp_chrome_folder) if re.match('evomap-.+\.txt',f)]
				#print (f"( Deleting {len(past_evofiles)} previous evomap files... )")
				for pevo in past_evofiles:
					os.remove(pevo)

				# take current population and mutate them
				for p in self.fi2.population:
					p.mutateMap(self.fi2.mapMutate)

				# evaluate the new population (using the MAP-Elites evaluator to retrieve the behavior characterstic)
				self.evaluate_pop(parallel)
				# for p in self.population:
				#     print(p)

				#print(f"pop: {[f'{e.con}-{e.fitness}' for e in self.population]}")

				#grab the "elite" and average fitness from the population
				sortedPop = sorted(self.fi2.population,reverse=True)
				bestPopFit = sortedPop[0].fitness
				avgPopFit = np.average([e.fitness for e in self.fi2.population])

				best_fit.append(bestPopFit)
				avg_fit.append(avgPopFit)

				#confirm sorting
				#print([f"{e.con}-{e.fitness}" for e in self.population])
				#print(f"sorted pop: {[f'{e.con}-{e.fitness}' for e in sortedPop]}")

				# sort into feasible and infeasible
				self.fi2.sortPop()

				# sort the population into matrices
				self.pop2Matrix()

				# based on the iteration number - export to files
				if outputArc and (i+1) % self.fi2.output_interval == 0:
					self.exportMAPElites(self.output_folder,f"Gen-{(i+1)}")

				# select a new population (USING MAP ELITES METHOD)
				self.fi2.population = self.newPop(popsize,self.eliteProb,self.fi2.feasRate)

				pbar.update(1)
				dm = self.divideMaps()
				pbar.set_description(f"#[e,f,i]: {[len(dm['elites']),len(dm['feas']),len(dm['infeas'])]} | Best: {bestPopFit:.3f} | Avg: {avgPopFit:.3f}")

		#export the best and averages as a csv in the archive folder
		with open(os.path.join(self.fi2.output_folder,'best_fit.csv'),'w+') as bf:
			c = csv.writer(bf)
			c.writerow(best_fit)

		with open(os.path.join(self.fi2.output_folder,'avg_fit.csv'),'w+') as af:
			c = csv.writer(af)
			c.writerow(avg_fit)

	# export the MAP-Elites to a folder of text files with the information set
	def exportMAPElites(self, folderLoc, label, expInfeas=True):
		
		# make the folder if it doesn't exist yet
		for f in ["ELITE","FEAS","INFEAS"]:
			os.makedirs(os.path.join(folderLoc,f), exist_ok=True)

		#get the divided sets of elite, feasible, and infeasible
		mapSets = self.divideMaps()

		#export the elites
		for i in range(len(mapSets['elites'])):
			fm = mapSets['elites'][i]
			fm.exportMap(os.path.join(folderLoc,"ELITE", f"[{label}]_{fm.bvec2str()}.txt"))

		# export the feasible population
		for i in range(len(mapSets['feas'])):
			fm = mapSets['feas'][i]
			fm.exportMap(os.path.join(folderLoc,"FEAS", f"{self.fi2.persona}-{i}-F_[{label}].txt"))

		if expInfeas:
			# export the infeasible population
			for i in range(len(mapSets['infeas'])):
				fm = mapSets['infeas'][i]
				fm.exportMap(os.path.join(folderLoc,"INFEAS", f"{self.fi2.persona}-{fm.bvec}-I_[{label}].txt"))




# run the main program if using this straight
if __name__ == "__main__":
	global INIT_ASCII_TILES, MUT_ASCII_TILES
	with open("config_mapelites.yml", "r") as f:
		config = yaml.safe_load(f)
	expset = MAPElites(config)

	#set choice tiles for initializing map and mutating map
	#INIT_ASCII_TILES = []
	#MUT_ASCII_TILES = []
	for t in config['INIT_ASCII_TILES']:
		INIT_ASCII_TILES.append(asc_char_map[t])
	for t in config['MUT_ASCII_TILES']:
		MUT_ASCII_TILES.append(asc_char_map[t])


	print(" -------------   MD2 MAP-ELITES EXPERIMENT   -------------")
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
	print(f"> MAP-ELITES DIMENSION SIZE: {config['MAP_ELITES_DIM']}")
	print(f"> MAP-ELITES MAX CELL:       {config['MAX_CELL_SIZE']}")
	print(f"> MAP-ELITES ELITE RATE:     {config['ELITE_PROB']}")
	print("")
	print("")

	expset.evolvePop(config["POPSIZE"], config["ITERATIONS"], parallel=config['PARALLEL'], outputArc=config['OUTPUT_ARC'])




