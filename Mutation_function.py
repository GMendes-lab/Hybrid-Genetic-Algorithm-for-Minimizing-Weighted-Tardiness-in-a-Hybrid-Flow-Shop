import copy
import random

def mutation(population, mutation_rate):
    """
    Applies mutation to a population of individuals.
    
    Args:
        population: List of individuals (each individual is a dictionary of machines)
        mutation_rate: Mutation rate (0.0 to 1.0) - percentage of population to be mutated
        
    Returns:
        New population with mutated individuals (deep copy of originals)
    """
    
    # Create deep copy of population to avoid modifying the original
    pop_mutated = copy.deepcopy(population)
    
    # Calculate number of individuals to mutate
    n_mutate = max(1, int(mutation_rate * len(pop_mutated)))
    
    # Select random individuals for mutation
    selected_indices = random.sample(range(len(pop_mutated)), n_mutate)
    
    for idx in selected_indices:
        ind = pop_mutated[idx]
        
        # Select machines to mutate (at least 1, maximum all)
        n_machines = random.randint(1, len(ind))
        machines_to_mutate = random.sample(list(ind.keys()), n_machines)
        
        for machine in machines_to_mutate:
            # Apply different types of mutation
            mutation_type = random.choice([
                'shuffle', 
                'swap', 
                'inversion',
                'scramble'
            ])
            
            jobs = ind[machine]
            
            if len(jobs) < 2:
                continue  # Doesn't make sense to mutate with less than 2 jobs
                
            if mutation_type == 'shuffle':
                # Shuffles all jobs
                random.shuffle(jobs)
                
            elif mutation_type == 'swap':
                # Swaps two random jobs
                pos1, pos2 = random.sample(range(len(jobs)), 2)
                jobs[pos1], jobs[pos2] = jobs[pos2], jobs[pos1]
                
            elif mutation_type == 'inversion':
                # Inverts a random subsequence
                start, end = sorted(random.sample(range(len(jobs)), 2))
                jobs[start:end+1] = jobs[start:end+1][::-1]
                
            elif mutation_type == 'scramble':
                # Shuffles a random subsequence
                start, end = sorted(random.sample(range(len(jobs)), 2))
                segment = jobs[start:end+1]
                random.shuffle(segment)
                jobs[start:end+1] = segment
    
    return pop_mutated