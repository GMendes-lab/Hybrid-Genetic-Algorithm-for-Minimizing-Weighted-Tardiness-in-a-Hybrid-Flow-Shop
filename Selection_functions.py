import random

def roulette_selection(individuals, selection_size, population_fitness):
    """
    Roulette Wheel Selection (Proportional to Fitness):
    1. Receives fitness values directly (population_fitness)
    2. Calculates selection probabilities proportional to fitness
    3. Randomly selects individuals based on these probabilities

    Args:
        individuals: List of individuals (any format)
        selection_size: Number of individuals to select
        population_fitness: List with fitness values corresponding to each individual

    Returns:
        List with selected individuals
    """
    # Sum total fitness
    total_fitness = 0
    for i in range(len(population_fitness)):
        total_fitness += population_fitness[i]

    # Avoid division by zero if total fitness is 0
    if total_fitness == 0:
        selection_prob = [1 / len(population_fitness)] * len(population_fitness)
    else:
        selection_prob = [fit / total_fitness for fit in population_fitness]

    selected = random.choices(individuals, weights=selection_prob, k=selection_size)

    return selected

def rank_selection(individuals, k, evolutionary_pressure, population_fitness):
    """
    Rank Selection:
    1. Sorts individuals by fitness (from best to worst)
    2. Assigns weights based on ranking with adjustable evolutionary pressure
    3. Randomly selects individuals based on ranking weights

    Args:
        individuals: List of individuals (any format)
        k: Number of individuals to select
        evolutionary_pressure: Selection intensity (1.0 = uniform, >1.0 favors the best)
        population_fitness: Optional list with corresponding fitness values

    Returns:
        List with selected individuals
    """
    # If provided, uses external fitness values
    if population_fitness is not None:
        # Pair individuals with their fitness for sorting
        paired = list(zip(individuals, population_fitness))
        # Sort by fitness (from best to worst)
        paired.sort(key=lambda x: x[1])
        sorted_inds = [item[0] for item in paired]
    else:
        # Assume individuals have fitness.values attribute (like in DEAP)
        sorted_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0])

    n = len(sorted_inds)

    # Weight calculation with evolutionary pressure
    if evolutionary_pressure == 1.0:
        # Uniform selection
        weights = [1.0 for _ in range(n)]
    else:
        # Linear formula for ranking weights
        weights = [(2 - evolutionary_pressure) + 2 * (evolutionary_pressure - 1) * (i / (n - 1))
                  for i in range(n)]

    # Select individuals
    selected = random.choices(sorted_inds, weights=weights, k=k)

    return selected

def tournament_selection(individuals, k, tournament_size, fitness_values, maximization=False):
    """
    Tournament Selection:
    1. For each selection, randomly chooses 'tournament_size' individuals
    2. Selects the best (or worst) of these individuals according to the criterion

    Args:
        individuals: List of individuals (any format)
        k: Number of individuals to select
        tournament_size: Number of competitors in each tournament (default=3)
        fitness_values: Optional list with corresponding fitness values
        maximization: True for maximization (selects the best), False for minimization

    Returns:
        List with selected individuals
    """
    # Check if tournament size is valid
    tournament_size = min(tournament_size, len(individuals))

    selected = []

    for _ in range(k):
        # Select random competitors
        competitors = random.sample(individuals, tournament_size)

        # Determine winner based on fitness
        if fitness_values is not None:
            # Use the provided fitness_values list
            competitors_fitness = [fitness_values[individuals.index(ind)] for ind in competitors]
            best_idx = competitors_fitness.index(max(competitors_fitness)) if maximization else competitors_fitness.index(min(competitors_fitness))
            winner = competitors[best_idx]
        else:
            # Assume individuals have fitness.values attribute (like in DEAP)
            if maximization:
                winner = max(competitors, key=lambda ind: ind.fitness.values[0])
            else:
                winner = min(competitors, key=lambda ind: ind.fitness.values[0])

        selected.append(winner)

    return selected