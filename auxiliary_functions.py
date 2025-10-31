import pandas as pd
from Parameters import *


def preprocess_raw_data(raw_df, schedule_date):
    """
    Preprocesses raw data, adding necessary columns and standardizing names.

    Args:
        raw_df: DataFrame with raw data
        schedule_date: Reference date for due date calculation

    Returns:
        Processed DataFrame
    """
    schedule_date_dt = pd.to_datetime(schedule_date)

    # Converting the 'numbers' column to string
    processed_df = raw_df.copy()
    processed_df['Due Date'] = pd.to_datetime(processed_df['Due Date'])
    processed_df['Delivery Time'] = (processed_df['Due Date'] - schedule_date_dt).dt.days
    processed_df = processed_df.rename(columns=lambda x: x.upper())
    processed_df = processed_df.rename(columns={processed_df.columns[0]: 'JOB'})
    processed_df['JOB'] = processed_df['JOB'].astype(str)

    return processed_df

def extract_gross_data(processed_df):
    """
    Extracts all essential data for production planning and scheduling in a single function.

    Args:
        processed_df: Processed DataFrame containing production data

    Returns:
        tuple: (jobs_list, production_goals, quantities_dict, workcenter_assignments, due_dates_dict, priority_weights_dict)
    """
    # Extraction of basic production data
    jobs_list = pd.array(processed_df['JOB'].unique())

    # Dictionaries for goals and quantities
    production_goals = {(row['JOB'], row['WORKCENTER']): row['GOAL'] for _, row in processed_df.iterrows()}
    quantities_dict = {(row['JOB'], row['WORKCENTER']): row['QUANTITY'] for _, row in processed_df.iterrows()}

    # Workcenter assignments
    workcenter_assignments = {}
    for _, row in processed_df.iterrows():
        wc = row['WORKCENTER']
        op = row['JOB']
        workcenter_assignments.setdefault(wc, set()).add(op)

    # Extraction of scheduling attributes
    valid_entries = processed_df[processed_df['DUE DATE'].notna()].copy()
    due_dates_dict = {row['JOB']: row['DELIVERY TIME'] for _, row in valid_entries.iterrows()}

    priority_weights_dict = (
        {row['JOB']: row['PRIORITY'] for _, row in valid_entries.iterrows()}
        if 'PRIORITY' in valid_entries.columns
        else {job: 1.0 for job in due_dates_dict.keys()}
    )

    return (
        jobs_list,
        production_goals,
        quantities_dict,
        workcenter_assignments,
        due_dates_dict,
        priority_weights_dict
    )

def setup_time(df_data, workcenters, dict_machines, dict_machine_turns, time_for_turn):
    """
    Generates a setup time dictionary for each (job_from, job_to, machine) combination in days,
    with specific rules for each workcenter type.

    Parameters:
    - df_data: DataFrame with columns JOB, WORKCENTER, PRODUCT
    - workcenters: List of workcenters to consider
    - dict_machines: Dictionary {workcenter: [machines]}
    - dict_machine_turns: Dictionary {machine: number of shifts}
    - time_for_turn: Hours per shift

    Returns:
    - {workcenter: {(job_from, job_to, machine): setup_time_in_days}}
    """
    # Define setup rules by workcenter type
    SETUP_RULES = {
        'PTH': {'same': 0.5, 'different': 1.0},    # 30 min (same), 1h (different)
        'SMT': {'same': 0.5, 'different': 1.0},     # 30 min (same), 1h (different)
        'PLASTIC': {'same': 0.5, 'different': 2.0}, # 30 min (same), 2h (different)
        'default': {'same': 0.25, 'different': 1.0}  # 15 min (same), 1h (different)
    }

    dict_setup_matrices = {}

    for wc in workcenters:
        # Determine which setup rule to use
        setup_rule = None
        for key in SETUP_RULES:
            if key in wc:  # Check if WC name contains the key (ex: 'PTH' in 'PTH_LINE1')
                setup_rule = SETUP_RULES[key]
                break

        if setup_rule is None:
            setup_rule = SETUP_RULES['default']
            print(f"Info: Using default rule for {wc}")

        # Initialize setup matrix for this workcenter
        dict_setup_matrices[wc] = {}

        # Filter jobs for current workcenter and remove duplicates
        wc_data = df_data[df_data['WORKCENTER'] == wc]
        unique_jobs = wc_data.drop_duplicates('JOB')

        if unique_jobs.empty:
            print(f"Warning: No jobs found for {wc}")
            continue

        # Create JOB → PRODUCT mapping
        job_to_product = dict(zip(unique_jobs['JOB'], unique_jobs['PRODUCT']))
        jobs = list(job_to_product.keys())

        # Get machines for this workcenter
        machines = dict_machines.get(wc, [])

        # Generate all possible combinations
        for job_from in jobs:
            for job_to in jobs:
                for machine in machines:
                    # Determine if products are the same
                    same_product = job_to_product[job_from] == job_to_product[job_to]

                    # Apply the correct setup rule
                    setup_hours = setup_rule['same'] if same_product else setup_rule['different']

                    # Convert to days considering machine shifts
                    setup_days = setup_hours / (dict_machine_turns.get(machine, 1) * time_for_turn)

                    # Store in dictionary
                    dict_setup_matrices[wc][(job_from, job_to, machine)] = setup_days

    return dict_setup_matrices

def processing_time(list_workcenters, dict_machines, dict_machine_turns, time_for_turn, quantity, goal, workcenter_to_ops):
    """
    Function responsible for calculating the processing time of each job, considering goal, shifts and machine

    Parameters:
    - list_workcenters: List of work centers
    - dict_machines: Dictionary {workcenter: [machines]}
    - dict_machine_turns: Dictionary {machine: number of shifts}
    - time_for_turn: Working time per shift (in hours)
    - quantity: Dictionary {job: quantity to produce}
    - goal: Dictionary {job: production goal per hour}

    Returns:
    - dict_processing_time = {(job, workcenter, machine): processing_time}
    """
    dict_processing_time = {}

    for workcenter in list_workcenters:
        # Get only the jobs for this workcenter
        workcenter_ops = workcenter_to_ops.get(workcenter, [])

        for machine in dict_machines.get(workcenter, []):
            for op in workcenter_ops:  # Only processes jobs for this workcenter
                    work_hours = quantity[(op, workcenter)] / goal[(op, workcenter)]
                    dict_processing_time[(op, workcenter, machine)] = work_hours / (dict_machine_turns.get(machine, 1) * time_for_turn)

    return dict_processing_time

def eligibility(df_data, workcenters, dict_machines):
    """
    Returns a dictionary in the format {(Job, Machine): 0 or 1}, where:
    - 1 = Eligible
    - 0 = Not eligible

    Rules:
    - ASSEMBLY: Job only runs on machines listed in MACHINE (comma separated).
    - Other workcenters:
        - "Mandatory": Job only runs on the machine specified in MACHINE.
        - "Preferential": Job runs on any machine in the workcenter.
        - Other values (ex: machine list): Job only runs on listed machines.
    """
    eligibility_dict = {}

    for wc in workcenters:
        machines = dict_machines.get(wc, [])
        wc_data = df_data[df_data['WORKCENTER'] == wc]

        for _, row in wc_data.iterrows():
            op = row['JOB']
            constraints = row.get('CONSTRAINTS', '')
            machine_info = row['MACHINE']

            # ASSEMBLY workcenter (specific rule)
            if wc == 'ASSEMBLY':
                allowed_machines = [m.strip() for m in machine_info.split(',')] if pd.notna(machine_info) else []
                for machine in machines:
                    eligibility_dict[(op, machine)] = 1 if machine in allowed_machines else 0

            # Other workcenters
            else:
                # Case 1: "Mandatory" constraint
                if constraints == "Mandatory":
                    required_machine = machine_info
                    for machine in machines:
                        eligibility_dict[(op, machine)] = 1 if machine == required_machine else 0

                # Case 2: "Preferential" constraint → all eligible
                elif constraints == "Preferential":
                    for machine in machines:
                        eligibility_dict[(op, machine)] = 1

                # Case 3: Other values (ex: machine list in CONSTRAINTS)
                elif pd.notna(constraints) and constraints != "":
                    allowed_machines = [m.strip() for m in constraints.split(',')]
                    for machine in machines:
                        eligibility_dict[(op, machine)] = 1 if machine in allowed_machines else 0

                # Case 4: Empty/null CONSTRAINTS → none eligible (or all, as needed)
                else:
                    for machine in machines:
                        eligibility_dict[(op, machine)] = 0  # Or 1, if default should be "all eligible"

    return eligibility_dict