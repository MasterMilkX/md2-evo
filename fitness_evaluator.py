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

def evaluate_fitness(result_obj, target_persona="R", func="success", cascading=True):
    assert(func in supported_fitness_functions, f"Input function {func} not part of supported fitness functions.")
    if func == "success":
        return eval_fitness_success(result_obj, target_persona, cascading)
    elif func == "gma":
        return eval_fitness_gma(result_obj, target_persona)

def eval_fitness_success(result_obj, target_persona, cascading):
    target_fitness = 0.0
    non_target_fitness = 0.0
    for persona in supported_personas:
        
        result_persona = result_obj[persona]
        level_report = result_persona["levelReport"]
        win_or_lose = int(level_report["exitUtility"])
        end_dist_to_exit = level_report["endDistToExit"]
        longest_path = level_report["longestPath"]
        health_left = level_report["health"]
        
        if persona == target_persona:
            if win_or_lose == 0:
                target_fitness = 1 * target_weight
            else:
                target_fitness+= (end_dist_to_exit / longest_path) * target_weight
        else:
            # divide in half to make room for 2 personas
            non_target_fitness += 0.5 * (1 - target_weight) * (10 - health_left)
        
    # cascading
    if cascading:
        if target_fitness != 0.5:
            fitness = target_fitness
        else:
    else:
        fitness = target_fitness + non_target_fitness

    return fitness


    

def eval_fitness_gma(result_obj, target_persona):
    # read in GMA data
    raise NotImplementedError("gma func not implemented yet")
