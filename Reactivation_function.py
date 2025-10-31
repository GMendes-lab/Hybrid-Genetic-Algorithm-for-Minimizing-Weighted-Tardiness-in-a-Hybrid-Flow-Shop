from EDD_functions import *

def reactivate_population(population, fitness, reactivation_percentage):
    """
    Replaces part of the population with new random individuals.
    
    Args:
        population: current list of individuals
        fitness: list of fitness values
        reactivation_percentage: fraction of population to be reactivated
        
    Returns:
        new_population: population with best individuals preserved and new random individuals added
    """
    pop_size = len(population)
    keep_count = int(pop_size * (1 - reactivation_percentage))

    # Sort population by fitness (lower is better)
    sorted_indices = np.argsort(fitness)

    # Keep the best individuals
    best_individuals = [population[i] for i in sorted_indices[:keep_count]]

    # Generate new individuals to replenish the population
    new_count = pop_size - keep_count
    new_individuals = generate_optimized_population(selected_orders, due_dates_dict, dict_processing_time, dict_eligibility, list_workcenters, dict_machines, dict_setup_matrices, 5, 100)

    # Combine the best with the new individuals
    new_population = best_individuals + new_individuals

    return new_population