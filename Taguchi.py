from auxiliary_functions import *
from EDD_functions import *
from Fitness_functions import *
from Selection_functions import *
from Crossover_functions import *
from Mutation_function import *
from Replacement_functions import *


#=== FUNCTION INVOCATION ===#

processed_df = preprocess_raw_data(raw_data_df, schedule_date)

jobs_list, production_goals, quantities_dict, workcenter_assignments, due_dates_dict, priority_weights_dict = extract_gross_data(processed_df)

# generate the processing time dictionary
dict_setup_matrices = setup_time(processed_df, list_workcenters, dict_machines, dict_machine_turns, time_for_turn)

dict_processing_time = processing_time(list_workcenters, dict_machines, dict_machine_turns, time_for_turn, quantities_dict, production_goals, workcenter_assignments)

dict_eligibility = eligibility(processed_df, list_workcenters, dict_machines)

def reactivate_population(population, fitness, reactivation_percentage, instance):
    """
    Replaces part of the population with new random individuals.

    - population: current list of individuals
    - fitness: list of fitness values
    - generate_individual: function that generates a new random individual
    - reactivation_percentage: fraction of population to be reactivated
    """
    pop_size = len(population)
    keep_count = int(pop_size * (1 - reactivation_percentage))

    # Sort population by fitness (lower is better)
    sorted_indices = np.argsort(fitness)

    # Keep the best individuals
    best_individuals = [population[i] for i in sorted_indices[:keep_count]]

    # Generate new individuals to replenish the population
    new_count = pop_size - keep_count
    new_individuals = generate_optimized_population(instance, due_dates_dict, dict_processing_time,
                                            dict_eligibility, list_workcenters, dict_machines,
                                            dict_setup_matrices, 5, pop_size//2)

    # Combine the best with the new ones
    new_population = best_individuals + new_individuals

    return new_population

#=== Taguchi ===#
# Generate and visualize the matrix
EXPERIMENTS =  [{'restart': True, 'popsize': 100, 'selection': 'tournament', 'crossover': 'OX', 'pmut': 0.02, 'replacement': 'hill_climbing', 'MaxGen': 500}]

# List of instance sizes
INSTANCE_SIZES = [500, 450, 400, 350, 300, 250, 200, 150, 140, 120, 100, 80, 40, 20]

# Generate 5 instances for each size
INSTANCES = []
for size in INSTANCE_SIZES:
    for i in range(5):
        # Assuming jobs_list is your original list of jobs
        instance = random.sample(jobs_list.tolist(), size)
        INSTANCES.append(instance)

TOURNAMENT_SIZE = 7
SELECTION_PRESSURE = 1.8
STAGNATION_LIMIT = 10
REACTIVATION_PERCENTAGE = 0.3
ELITISM = 10

def Taguchi(instance):
    temperature = 100
    time_limit = 60*60
    
    # DataFrames to store all results
    final_results = pd.DataFrame(columns=['experiment', 'instance_size', 'best_fitness', 'ARP', 'Time'])
    complete_statistics = pd.DataFrame()
    
    for config in EXPERIMENTS:
        # Initialization of metrics
        best_fitness = float('inf')
        first_fitness_better = None
        best_individual = None
        generations_without_improvement = 0
        start_time = time.time()
        elapsed = 0
        
        # DataFrame for statistics per generation of this experiment
        experiment_statistics = pd.DataFrame()
        
        # 1. Population initialization
        population = generate_optimized_population(
            instance, due_dates_dict, dict_processing_time,
            dict_eligibility, list_workcenters, dict_machines,
            dict_setup_matrices, 5, config['popsize'])

        # 2. Evolutionary loop
        for gen in range(config['MaxGen']):
            if elapsed > time_limit:
                print("Time limit reached")
                break
                
            # Evaluation
            population_fitness = calculate_fitness_population(
                population, dict_processing_time,
                dict_setup_matrices, due_dates_dict,
                priority_weights_dict, buffer_time
            )

            # Update best fitness
            current_best_idx = np.argmin(population_fitness)
            current_best = population_fitness[current_best_idx]
            
            if gen == 0:
                first_fitness_better = current_best
                
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = deepcopy(population[current_best_idx])
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Reactivation
            if config['restart'] and generations_without_improvement >= STAGNATION_LIMIT:
                population = reactivate_population(population, population_fitness, 0.5, instance)
                generations_without_improvement = 0

            # Selection
            if config['selection'] == 'tournament':
                selected_parents = tournament_selection(population, len(population), TOURNAMENT_SIZE, population_fitness)
            elif config['selection'] == 'roulette':
                selected_parents = roulette_selection(population, len(population), population_fitness)
            else:
                selected_parents = rank_selection(population, len(population), SELECTION_PRESSURE, population_fitness)

            fitness_selected = calculate_fitness_population(
                selected_parents, dict_processing_time,
                dict_setup_matrices, due_dates_dict,
                priority_weights_dict, buffer_time
            )

            # Crossover
            if config['crossover'] == 'OX':
                offspring = ox_crossover(selected_parents, dict_eligibility, instance, len(population))
            else:
                offspring = []
                for i in range(0, len(selected_parents)-1, 2):
                    child1 = pmx_crossover(selected_parents[i], selected_parents[i+1], dict_eligibility, instance)
                    child2 = pmx_crossover(selected_parents[i], selected_parents[i+1], dict_eligibility, instance)
                    offspring.extend([child1, child2])

            # Mutation
            offspring = mutation(offspring, config['pmut'])

            # Offspring evaluation
            fitness_offspring = calculate_fitness_population(
                offspring, dict_processing_time,
                dict_setup_matrices, due_dates_dict,
                priority_weights_dict, buffer_time
            )
            
            # Replacement
            if config['replacement'] == 'simple':
                population = simple_replacement(population, offspring, population_fitness, fitness_offspring)
            elif config['replacement'] == 'SA':
                population = simulated_annealing_substitution(population, offspring, population_fitness, fitness_offspring, temperature, ELITISM)
                temperature = min(temperature * 0.95, 1)
            else:
                population = hill_climbing_substitution(population, offspring, population_fitness, fitness_offspring, ELITISM)

            # ARP calculation
            if first_fitness_better == 0:
                ARP = "too big"
            else:
                ARP = ((first_fitness_better - current_best)/first_fitness_better)*100
            
            # Time update
            elapsed = time.time() - start_time
            
            # Add current generation statistics
            generation_statistics = pd.DataFrame({
                'experiment': [str(config)],
                'generation': [gen+1],
                'Population Average': [np.average(population_fitness)],
                'Selected Average': [np.average(fitness_selected)],
                'Offspring Average': [np.average(fitness_offspring)],
                'best_fitness': [best_fitness],
                'execution_time': [elapsed],
                'diversity': [np.std(population_fitness)],
                'ARP': [ARP]
            })
            
            experiment_statistics = pd.concat([experiment_statistics, generation_statistics], ignore_index=True)
            
            print(f"Generation {gen+1}, test {EXPERIMENTS.index(config)+1}, ARP {ARP}%, best: {best_fitness}, time {elapsed}")
            
        # At the end of the experiment, add final results
        experiment_results = pd.DataFrame({
            'experiment': [str(config)],
            'instance_size': [len(instance)],
            'best_fitness': [best_fitness],
            'ARP': [ARP],
            'Time': [elapsed]
        })
        
        final_results = pd.concat([final_results, experiment_results], ignore_index=True)
        complete_statistics = pd.concat([complete_statistics, experiment_statistics], ignore_index=True)

    return final_results, complete_statistics

i = 0
# For all instances
complete_results = []
for instance in INSTANCES:
    taguchi_results, generational_results = Taguchi(instance)
    i += 1
    print(i)

    path = fr".xlsx"

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        taguchi_results.to_excel(writer, sheet_name='taguchi_results', index=False)
        generational_results.to_excel(writer, sheet_name='generational_results', index=False)
