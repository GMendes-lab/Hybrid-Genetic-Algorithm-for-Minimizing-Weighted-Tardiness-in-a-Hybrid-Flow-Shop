import random
from auxiliary_functions import *

def allocation(list_jobs, due_dates, processing_times, eligibility, machines):
    """
    Allocates jobs to machines following:
    1. Sorting by Earliest Due Date (EDD)
    2. Chooses eligible machine with shortest processing time
    3. Random tie-breaker for equal times

    Parameters:
        list_jobs: List of job IDs
        due_dates: Dictionary {job_id: due_date}
        processing_times: Dictionary {(job_id, wc, machine): time}
        eligibility: Dictionary {(job_id, machine): 1/0}
        machines: Dictionary {wc: [machines]}

    Returns:
        {(workcenter, machine): [job_sequence]}
    """
    # 1. Initialization with new format
    schedule = {}

    # 2. Sort jobs by EDD (converting to integers)
    try:
        jobs_sorted = sorted([job for job in list_jobs],
                            key=lambda x: due_dates[x])
    except (ValueError, KeyError) as e:
        raise ValueError(f"Error sorting jobs: {str(e)}")

    # 3. Fixed order of workcenters in production flow
    workcenter_flow = ['PLASTIC', 'SMT', 'PTH', 'ASSEMBLY']

    # 4. Allocation by flow stage
    for wc in workcenter_flow:
        if wc not in machines:
            continue

        # Initialize machines for this workcenter
        for machine in machines[wc]:
            schedule[(wc, machine)] = []

        for job in jobs_sorted:
            # 4.1. Eligible machines for this job
            eligible_machines = [
                m for m in machines[wc]
                if eligibility.get((job, m), 0) == 1
            ]

            if not eligible_machines:
                continue  # Job cannot be processed at this stage

            # 4.2. Find machine with shortest processing time
            min_time = float('inf')
            candidates = []

            for machine in eligible_machines:
                try:
                    time = processing_times[(job, wc, machine)]
                    if time < min_time:
                        min_time = time
                        candidates = [machine]
                    elif time == min_time:
                        candidates.append(machine)
                except KeyError:
                    continue

            if not candidates:
                continue  # No processing time found

            # 4.3. Random tie-breaker if needed
            chosen_machine = random.choice(candidates) if len(candidates) > 1 else candidates[0]

            # 4.4. Allocate job to chosen machine
            schedule[(wc, chosen_machine)].append(job)

    return schedule

def optimize_sequence_with_setup(schedule, dict_setup_matrices, lookahead):
    """
    Optimizes production sequence by minimizing setup times between consecutive operations.

    Args:
        schedule: {(workcenter, machine): [op1, op2, ...]}
        dict_setup_matrices: Dictionary of setup matrices by workcenter
        lookahead: Number of operations ahead to consider for reordering

    Returns:
        New optimized schedule in format {(workcenter, machine): [operations]}
    """
    optimized_schedule = {}

    # 1. Process each machine in original schedule
    for (wc, machine), sequence in schedule.items():
        current_sequence = sequence.copy()
        optimized_sequence = []

        while current_sequence:
            if not optimized_sequence:
                # First operation: take next available
                current_op = current_sequence.pop(0)
                optimized_sequence.append(current_op)
            else:
                # 2. Get last operation in optimized sequence
                last_op = optimized_sequence[-1]

                # 3. Analyze next 'lookahead' operations
                candidates = current_sequence[:lookahead]

                if not candidates:
                    break

                # 4. Find operation with minimum setup time
                best_candidate = None
                best_index = 0
                min_setup = float('inf')

                for i, candidate in enumerate(candidates):
                    setup_time = dict_setup_matrices.get(wc, {}).get(
                        (last_op, candidate, machine),
                        float('inf')  # Default if not found
                    )

                    if setup_time < min_setup:
                        min_setup = setup_time
                        best_candidate = candidate
                        best_index = i

                # 5. Add best candidate found
                if best_candidate is not None:
                    current_sequence.pop(best_index)
                    optimized_sequence.append(best_candidate)
                else:
                    # If no setup time defined, take next normal one
                    optimized_sequence.append(current_sequence.pop(0))

        # 6. Store optimized sequence
        optimized_schedule[(wc, machine)] = optimized_sequence

    return optimized_schedule

def generate_optimized_population(list_jobs, dict_due_dates, dict_processing_time, eligibility_dict, list_workcenters, dict_machines, dict_setup_matrices, optimization_passes, population_size):
    """
    Generates a population of optimized schedules

    Parameters:
    - n_iterations: Maximum number of iterations
    - population_size: Desired population size (None to use n_iterations)
    - ... (other parameters according to your original implementation)

    Returns:
    - List containing only optimized schedules
    """
    population = []

    # Version with assignment
    iterations = 0  # Default value
    if population_size is not None:
        iterations = population_size  # Or some calculation based on population_size

    for i in range(iterations):
        try:
            # 1. Generate initial schedule
            current_jobs = list_jobs

            # Add random variation after first iteration
            if i > 0:
                random.shuffle(current_jobs)

            schedule = allocation(current_jobs, dict_due_dates, dict_processing_time,
                                eligibility_dict, list_workcenters, dict_machines)

            # 2. Optimize schedule
            optimized = optimize_sequence_with_setup(schedule, dict_setup_matrices, optimization_passes)

            # 3. Add only optimized schedule to population
            population.append(optimized)

            # Check if reached desired size
            if population_size is not None and len(population) >= population_size:
                break

        except Exception as e:
            print(f"Error in iteration {i}: {str(e)}")
            continue

    return population