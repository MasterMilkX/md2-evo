from unittest import result
from utility import entr_fitness
import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

supported_fitness_functions = [
    "gma",
    "success"
]
supported_personas = [
    "R",
    "TC",
    "MK"
]

target_weight = 0.5

import_mechanics = ['BlobCombine',
 'BlobHit',
 'BlobPotion',
 'CollectPotion',
 'CollectTreasure',
 'Die',
 'GoblinMeleeHit',
 'GoblinRangedHit',
 'JavelinThrow',
 'MinitaurHit',
 'OgreHit',
 'OgreTreasure',
 'ReachStairs',
 'TriggerTrap',
 'UsePortal',
 'EndTurn',
 'EnemyKill'
]
y_gma = pd.read_csv("agent_y_import.csv")
y_gma = y_gma.set_index("Unnamed: 0")
agent_frequencies = pd.read_csv("n_agent_data.csv")
agent_frequencies = agent_frequencies.set_index("Unnamed: 0")

def evaluate_fitness(result_obj, asc_map, target_persona="R", func="success", cascading=True):
    assert(func in supported_fitness_functions,
           f"Input function {func} not part of supported fitness functions.")
    if func == "success":
        return eval_fitness_success(result_obj, asc_map, target_persona, cascading)
    elif func == "gma":
        return eval_fitness_gma(result_obj, asc_map, target_persona)

def eval_fitness_entropy(asc_map):
    asc_map = np.array(asc_map)
    return entr_fitness(asc_map)

def eval_fitness_success(result_obj, asc_map, target_persona, cascading):
    target_fitness = 0.0
    non_target_fitness = 0.0
    for persona in supported_personas:

        result_persona = result_obj[persona]
        level_report = result_persona["levelReport"]
        avgWin = result_persona["avgWin"]
        avgDist = result_persona["avgDist"]
        longest_path = level_report["longestPath"]
        avgHealth = result_persona["avgHealth"]

        if persona == target_persona:
            if avgWin == 1.0:
                target_fitness = 1
            else:
                target_fitness += (avgDist /
                                   longest_path)
        else:
            # divide in half to make room for 2 personas
            if avgHealth < 0:
                avgHealth = 0
            non_target_fitness +=  (10 - avgHealth)
    non_target_fitness = non_target_fitness / 20
    # cascading
    if cascading:
        if target_fitness != 1:
            fitness = target_fitness
        elif non_target_fitness != 1:
            fitness = target_fitness * target_weight+ non_target_fitness * (1 - target_weight)
        else:
            fitness = target_fitness * target_weight + non_target_fitness * (1-target_weight) + eval_fitness_entropy(asc_map)
    else:
        fitness = target_fitness * target_weight+ non_target_fitness * (1 - target_weight)

    return fitness


def eval_fitness_gma(result_obj, asc_map, target_persona):
    # read in GMA data
    target = y_gma.loc[target_persona]

    frequencies = {}
    for per in supported_personas:
        frequencies[per] = {}
        for mech in import_mechanics:
            frequencies[per][mech] = 0
            if mech in result_obj[per]["frequencies"].keys():
                frequencies[per][mech] = result_obj[per]["frequencies"][mech]
        frequencies[per]["Label"] = f"{per}-evo"
    frequencies_df = pd.DataFrame.from_dict(frequencies, orient='index')
    all_data = pd.concat([agent_frequencies, frequencies_df])
    all_data_n = all_data.copy()
    for col in all_data.columns:
        if col == "Label" or col == "IsUser" or col == "MapNumber":
            continue
        all_data_n[col] /= 1.0 * all_data[col].max()
    print("here")







def calc_mech_importance(fil_data, all_data, mech):
    fil_array = fil_data[mech].to_numpy()
    all_array = all_data[mech].to_numpy()
    sign = np.sign(fil_array.mean() - all_array.mean())
    if sign == 0:
        sign = 1
    value = wasserstein_distance(fil_array, all_array)
    return sign * value

def calc_mech_axis(data, values, mechanics):
    value_temp = {}
    row_index = []
    for mech in mechanics:
        value_temp[mech] = []
        for v in values:
            if v in data.index:
                traces = data.loc[v]
            else:
                continue
            if len(traces) > 0:
                value_temp[mech].append(calc_mech_importance(traces, data, mech))
                if v not in row_index:
                    row_index.append(v)
    return pd.DataFrame(value_temp, index=row_index) 
