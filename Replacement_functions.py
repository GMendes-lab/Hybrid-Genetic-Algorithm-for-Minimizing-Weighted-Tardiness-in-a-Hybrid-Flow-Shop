import random
import math

#@timeit
def simple_replacement(original_population, offspring, original_fitness, offspring_fitness):
    """
    Simple replacement that:
    1. Sorts the original population by fitness
    2. Eliminates the half with worst fitness
    3. Adds all offspring individuals
    4. Returns new population with original size

    Args:
        original_population: List of current individuals
        offspring: List of new generated individuals
        original_fitness: List of fitness values corresponding to original population

    Returns:
        New population composed of best half of original + all offspring
    """
    # 1. Pair individuals with their fitness for sorting
    population_with_fitness = list(zip(original_population, original_fitness))
    offspring_with_fitness = list(zip(offspring, offspring_fitness))

    # 2. Sort from best (lowest fitness) to worst (highest fitness)
    population_with_fitness.sort(key=lambda x: x[1])
    offspring_with_fitness.sort(key=lambda x: x[1])

    # 3. Select the best half
    half_pop = len(original_population) // 2
    best_pop = [ind for ind, fit in population_with_fitness[:half_pop]]

    half_off = len(offspring_with_fitness) // 2
    best_off = [ind for ind, fit in offspring_with_fitness[:half_off]]

    # 4. Combine best from previous generation with all offspring
    new_population = best_pop + best_off

    # 5. Ensure it doesn't exceed original size

    return new_population


def simulated_annealing_substitution(original_population, offspring, original_fitness, offspring_fitness, temperature, n_elite):
    """
    Simulated Annealing replacement with elitism that:
    1. For each pair of original individual and offspring, decides replacement
    2. Accepts worse solutions with probability that decreases with temperature
    3. Preserves the n best individuals from original population
    4. Returns new population with same size as original

    Args:
        original_population: List of current individuals
        offspring: List of new generated individuals
        original_fitness: List of fitness values corresponding to original population
        offspring_fitness: List of fitness values corresponding to offspring
        temperature: Current temperature value for probability calculation
        n_elite: Number of elite individuals to preserve (default: 10)

    Returns:
        New population with replacements based on simulated annealing criterion and elitism
    """
    # 1. Select the n best individuals from original population (elite)
    population_with_fitness = list(zip(original_population, original_fitness))

    population_with_fitness.sort(key=lambda x: x[1])

    elite_pop = [ind for ind, fit in population_with_fitness[:n_elite]]
    
    # 2. Apply simulated annealing for replacement
    new_population = []
    
    for i in range(len(original_population)):
        parent = original_population[i]
        child = offspring[i]
        fit_parent = original_fitness[i]
        fit_child = offspring_fitness[i]

        if fit_child < fit_parent:
            new_population.append(child)
        else:
            delta = fit_child - fit_parent
            prob = math.exp(-delta / temperature)
            if random.random() < prob:
                new_population.append(child)
            else:
                new_population.append(parent)
    
    # 3. Add elite individuals to new population
    new_population = elite_pop + new_population

    # Scale to original size
    new_population = new_population[:len(original_population)]
    
    return new_population

def hill_climbing_substitution(original_population, offspring, original_fitness, offspring_fitness, n_elite):
    """
    Hill Climbing replacement that:
    1. For each pair of original individual and offspring, decides replacement
    2. Accepts only better or equal solutions
    3. Returns new population with same size as original

    Args:
        original_population: List of current individuals
        offspring: List of new generated individuals
        original_fitness: List of fitness values corresponding to original population
        offspring_fitness: List of fitness values corresponding to offspring

    Returns:
        New population with replacements based on hill climbing criterion
    """
    # 1. Select the n best individuals from original population (elite)
    population_with_fitness = list(zip(original_population, original_fitness))

    population_with_fitness.sort(key=lambda x: x[1])

    elite_pop = [ind for ind, fit in population_with_fitness[:n_elite]]

    new_population = []
    
    for i in range(len(original_population)):
        parent = original_population[i]
        child = offspring[i]
        fit_parent = original_fitness[i]
        fit_child = offspring_fitness[i]

        # If child is better or equal, replace
        if fit_child <= fit_parent:
            new_population.append(child)
        else:
            new_population.append(parent)

    # New population with elite
    new_population = elite_pop + new_population

    # Scale to original size
    new_population = new_population[:len(original_population)]

    return new_population