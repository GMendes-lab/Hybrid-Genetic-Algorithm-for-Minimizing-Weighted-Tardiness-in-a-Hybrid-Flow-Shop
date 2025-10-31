from auxiliary_functions import *
from EDD_functions import *
from Fitness_functions import *
from Selection_functions import *
from Crossover_functions import *
from Mutation_function import *
from Replacement_functions import *
from Reactivation_function import *

#=== Calling Functions ===#

processed_df = preprocess_raw_data(raw_data_df, schedule_date)

jobs_list, production_goals, quantities_dict, workcenter_assignments, due_dates_dict, priority_weights_dict = extract_gross_data(processed_df)

# generate the setup time dictionary
dict_setup_matrices = setup_time(processed_df, list_workcenters, dict_machines, dict_machine_turns, time_for_turn)

# generate the processing time dictionary
dict_processing_time = processing_time(list_workcenters, dict_machines, dict_machine_turns, time_for_turn, quantities_dict, production_goals, workcenter_assignments)

# generate the eligibility dictionary
dict_eligibility = eligibility(processed_df, list_workcenters, dict_machines)


#=== Parameters ===#
TOURNAMENT_SIZE = 5
SELECTION_PRESSURE = 1.2
STAGNATION_LIMIT = 20
REACTIVATION_PERCENTAGE = 0.3
GENERATIONS = 100

#=== GENETIC ALGORITHM ===#

# Initialization

population = generate_optimized_population(jobs_list, due_dates_dict, dict_processing_time, dict_eligibility, list_workcenters, dict_machines, dict_setup_matrices, 5, 100)

best_individual = None
best_fitness = float('inf')
history = {
    'best_fitness': [],
    'avg_fitness': [],
    'diversity': []
}

for gen in range(GENERATIONS):
    # 1. Evaluate population fitness
    population_fitness = calculate_fitness_population(
        population,
        dict_processing_time,
        dict_setup_matrices,
        due_dates_dict,
        priority_weights_dict,
        buffer_time
    )

    # Update best individual
    current_best_idx = np.argmin(population_fitness)
    if population_fitness[current_best_idx] < best_fitness:
        best_fitness = population_fitness[current_best_idx]
        best_individual = deepcopy(population[current_best_idx])
        generations_without_improvement = 0
    else:
        generations_without_improvement += 1

    # Record metrics
    history['best_fitness'].append(best_fitness)
    history['avg_fitness'].append(np.mean(population_fitness))

    # 2. Tournament Selection
    selected_parents = rank_selection(
        population,
        len(population) // 2, 1.8,
        population_fitness)

    # 3. Crossover
    offspring = ox_crossover(selected_parents, dict_eligibility, jobs_list, 75)

    # 4. Mutation
    offspring = mutation(offspring, 0.01)

    # 5. Evaluate offspring
    offspring_fitness = calculate_fitness_population(
        offspring,
        dict_processing_time,
        dict_setup_matrices,
        due_dates_dict,
        priority_weights_dict,
        buffer_time
    )

    # 6. Replacement with elitism
    population = simple_replacement(
        population,
        offspring,
        population_fitness,
        offspring_fitness
    )
    
    # Reactivation
    if generations_without_improvement >= 20:
        population = reactivate_population(population, population_fitness, 0.5)
        generations_without_improvement = 0
        print("Reactivation Performed")

    # Save to appendix
    print(f"Gen {gen}: Best={best_fitness:.2f}, Avg={np.mean(population_fitness):.2f}")
    print(f"Genetic diversity {np.std(population_fitness):.2f}")