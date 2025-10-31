def calculate_completion_time(
    population_item, # Receives population[i] = {'individual': { (wc, machine): [operations] }}
    dict_processing_time,
    dict_setup_matrices,
    buffer_pth_assembly
):
    # Extracts the allocation dictionary of the individual
    individual = population_item

    # Initializes timelines for each machine
    timelines = {}
    for (wc, machine), ops in individual.items():
        timelines[(wc, machine)] = []

    # Dictionary to track completion of each operation in each workcenter
    operation_completion = {}

    # Workcenters in precedence order
    workcenter_order = ['PLASTIC', 'SMT', 'PTH', 'ASSEMBLY']

    # Process each workcenter in correct order
    for wc in workcenter_order:
        # PTH only starts after the maximum time between Plastic and SMT
        if wc == 'PTH':
            for op in operation_completion:
                if 'PLASTIC' in operation_completion[op] and 'SMT' in operation_completion[op]:
                    operation_completion[op]['PTH_PRECEDENCE'] = max(
                        operation_completion[op]['PLASTIC'],
                        operation_completion[op]['SMT']
                    )

        # Assembly only starts after PTH + buffer
        elif wc == 'ASSEMBLY':
            for op in operation_completion:
                if 'PTH' in operation_completion[op]:
                    operation_completion[op]['ASSEMBLY_PRECEDENCE'] = (
                        operation_completion[op]['PTH'] + buffer_pth_assembly
                    )

        # Process each machine in current workcenter
        for (wc_machine, machine), ops in individual.items():
            if wc_machine != wc:
                continue  # Only process current workcenter

            for i, op in enumerate(ops):
                # Processing time of operation on current machine
                processing_time = dict_processing_time.get((op, wc, machine), 0)

                # Setup (time between previous and current operation on same machine)
                setup = 0
                if i > 0:
                    previous_op = ops[i - 1]
                    setup = dict_setup_matrices[wc].get((previous_op, op, machine), 0)

                # Calculate start_time:
                # 1. Completion time of previous operation on SAME machine + setup
                machine_start_time = 0
                if i > 0:
                    last_op = timelines[(wc, machine)][-1]
                    machine_start_time = last_op['end'] + setup

                # 2. Completion time of operation in PREVIOUS WORKCENTER (if exists)
                precedence_start_time = 0
                if wc == 'SMT' or wc == 'PLASTIC':
                    pass  # No precedence (run in parallel)
                elif wc == 'PTH':
                    if op in operation_completion and 'PTH_PRECEDENCE' in operation_completion[op]:
                        precedence_start_time = operation_completion[op]['PTH_PRECEDENCE']
                elif wc == 'ASSEMBLY':
                    if op in operation_completion and 'ASSEMBLY_PRECEDENCE' in operation_completion[op]:
                        precedence_start_time = operation_completion[op]['ASSEMBLY_PRECEDENCE']

                # Start_time is the maximum between the two
                start_time = max(machine_start_time, precedence_start_time)

                # Calculate end_time
                end_time = start_time + processing_time

                # Update machine timeline
                timelines[(wc, machine)].append({
                    'OP': op,
                    'start': start_time,
                    'end': end_time
                })

                # Update operation completion in current workcenter
                if op not in operation_completion:
                    operation_completion[op] = {}
                operation_completion[op][wc] = end_time

    return timelines

def calculate_fitness_population(
    population,
    dict_processing_time,
    dict_setup_matrices,
    dict_due_dates,
    dict_weights,
    buffer_pth_assembly
):
    all_fitness = []

    for individual in population:
        # 1. Calculate completion times for the individual
        timeline = calculate_completion_time(
            individual,
            dict_processing_time,
            dict_setup_matrices,
            buffer_pth_assembly
        )

        # 2. Filter only 'ASSEMBLY' sector
        assembly_ops = [
            op for (wc, machine), ops in timeline.items()
            if wc == 'ASSEMBLY'
            for op in ops
        ]

        # 3. Calculate weighted tardiness for each operation in Assembly
        total_weighted_tardiness = 0
        for op in assembly_ops:
            op_id = op['OP']
            end_time = op['end']
            due_date = dict_due_dates.get(op_id, 0)  # If no due_date, assume 0
            weight = dict_weights.get(op_id, 1.0)    # Default weight = 1.0

            tardiness = max(0, end_time - due_date)
            total_weighted_tardiness += tardiness * weight

        # 4. Add to fitness list (lower is better)
        all_fitness.append(total_weighted_tardiness)

    return all_fitness