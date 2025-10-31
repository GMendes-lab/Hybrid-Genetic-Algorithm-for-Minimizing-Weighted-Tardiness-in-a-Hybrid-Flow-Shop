import random
from copy import deepcopy
from collections import Counter, defaultdict

def ox_crossover(selection, dict_eligibility, jobs_list, offspring_size):
    """
    Generates a population of offspring using OX crossover between selected parents.
    
    Args:
        selection: List of available parents for selection
        dict_eligibility: Eligibility dictionary in format {(job, machine): 1/0}
        jobs_list: List of all jobs that should be considered
        offspring_size: Number of offspring to generate
        
    Returns:
        List of generated offspring
    """
    
    # Preprocessing eligibility dictionary
    preprocessed_eligibility = {}
    for (job, machine), eligible in dict_eligibility.items():
        if eligible == 1:
            preprocessed_eligibility.setdefault(job, []).append(machine)
    
    pop_offspring = []
    
    for _ in range(offspring_size):
        # Random selection of different parents
        parent1, parent2 = random.sample(selection, 2)
        
        # Selection of eligible machines for crossover
        valid_machines = [
            k for k in parent1
            if k in parent2 and len(parent1[k]) >= 2 and len(parent2[k]) >= 2
        ]
        
        # Create offspring with crossover
        offspring = {k: v.copy() for k, v in parent1.items()}
        
        # Apply crossover only if there are valid machines
        if valid_machines:
            n_tuples = random.randint(1, max(1, len(valid_machines)-1))
            selected_tuples = random.sample(valid_machines, n_tuples)
            
            for (workcenter, machine) in selected_tuples:
                # Random cut point
                c1 = random.randint(1, len(parent1[(workcenter, machine)]) - 1)
                offspring[(workcenter, machine)][c1:] = parent2[(workcenter, machine)][c1:]
                
                # Process workcenter to remove duplicates
                workcenter_machines = [k for k in parent1 if k[0] == workcenter]
                jobs_in_workcenter = set()
                
                for (wc, machine) in workcenter_machines:
                    jobs_on_machine = offspring[(wc, machine)].copy()
                    new_jobs = []
                    
                    for job in jobs_on_machine:
                        if job not in jobs_in_workcenter:
                            new_jobs.append(job)
                            jobs_in_workcenter.add(job)
                    
                    offspring[(wc, machine)] = new_jobs
                
                # Allocate missing jobs
                missing_jobs = [job for job in jobs_list if job not in jobs_in_workcenter]
                
                for job in missing_jobs:
                    eligible_machines = preprocessed_eligibility.get(job, [])
                    available_machines = [
                        machine for machine in eligible_machines 
                        if (workcenter, machine) in offspring
                    ]
                    
                    if available_machines:
                        # Choose machine with least load
                        allocation_machine = min(
                            available_machines,
                            key=lambda machine: len(offspring[(workcenter, machine)])
                        )
                        offspring[(workcenter, allocation_machine)].append(job)
        
        pop_offspring.append(offspring)
    
    return pop_offspring

def pmx_crossover(parent_1, parent_2, eligibility_dict, jobs_list, n_tuples=2, verbose=False):
    """
    Optimized PMX crossover version for hierarchical scheduling.
    
    Args:
        parent_1, parent_2: Dictionaries {(wc, machine): [jobs]}
        eligibility_dict: {(job, machine): 1 or 0}
        jobs_list: List of all jobs
        n_tuples: Number of tuples for crossover
        verbose: Detailed logging mode
        
    Returns:
        Dictionary with generated child
    """
    # 1. Initial preprocessing
    child = {k: v.copy() for k, v in parent_2.items()}
    expected_jobs = set(map(int, jobs_list))
    
    # 2. Optimized selection of tuples for crossover
    common_tuples = [k for k in parent_1 if k in parent_2]
    selected_tuples = random.sample(common_tuples, min(n_tuples, len(common_tuples))) if common_tuples else []
    
    if verbose:
        print(f"Tuples selected for crossover: {selected_tuples}")

    # 3. Processing selected tuples
    for tuple_key in selected_tuples:
        parent_jobs = parent_1[tuple_key]
        if len(parent_jobs) < 2:
            continue
            
        # Selection of cut points
        c1, c2 = sorted(random.sample(range(len(parent_jobs)), 2))
        
        # PMX crossover application
        child[tuple_key][c1:c2+1] = parent_jobs[c1:c2+1]
        
        # Treatment of repeated jobs
        current_jobs = child[tuple_key]
        counter = Counter(current_jobs)
        
        # Optimized replacement of duplicates
        if len(counter) < len(current_jobs):
            available_jobs = [j for j in parent_jobs if j not in current_jobs]
            if available_jobs:
                for job, count in counter.items():
                    if count > 1 and available_jobs:
                        idx = current_jobs.index(job)
                        current_jobs[idx] = available_jobs.pop()

    # 4. Optimized duplicate removal
    wc_jobs = defaultdict(set)
    
    # First pass: identify existing allocations
    for (wc, machine), jobs in child.items():
        unique_jobs = []
        for job in jobs:
            if job not in wc_jobs[wc]:
                unique_jobs.append(job)
                wc_jobs[wc].add(job)
        child[(wc, machine)] = unique_jobs

    # 5. Optimized allocation of missing jobs
    # Pre-computation of eligible machines per job
    job_machines = defaultdict(list)
    for (job, machine), eligible in eligibility_dict.items():
        if eligible == 1:
            job_machines[job].append(machine)
    
    # Efficient allocation
    for wc in set(wc for wc, _ in child.keys()):
        # Expected jobs in this WC
        wc_jobs_expected = {j for j in expected_jobs 
                           if any((wc, m) in child for m in job_machines.get(j, []))}
        
        # Missing jobs
        missing = wc_jobs_expected - wc_jobs[wc]
        
        for job in missing:
            eligible_machines = [m for m in job_machines.get(job, []) 
                               if (wc, m) in child]
            
            if eligible_machines:
                # Select machine with least load
                target_machine = min(eligible_machines, 
                                   key=lambda m: len(child.get((wc, m), [])))
                
                child.setdefault((wc, target_machine), []).append(job)
                wc_jobs[wc].add(job)
                
                if verbose:
                    print(f"Allocated job {job} on machine {target_machine}")
            elif verbose:
                print(f"WARNING: Job {job} without eligible machines in WC {wc}")

    return child