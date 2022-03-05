from utility import entr_fitness
import numpy as np
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
        exit_utility = int(level_report["exitUtility"]) 
        alive = level_report["alive"]
        end_dist_to_exit = level_report["endDistToExit"]
        longest_path = level_report["longestPath"]
        health_left = level_report["health"]

        if persona == target_persona:
            if exit_utility == 1.0 and alive:
                target_fitness = 1
            else:
                target_fitness += (end_dist_to_exit /
                                   longest_path)
        else:
            # divide in half to make room for 2 personas
            if health_left < 0:
                health_left = 0
            non_target_fitness +=  (10 - health_left)
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
    raise NotImplementedError("gma func not implemented yet")
