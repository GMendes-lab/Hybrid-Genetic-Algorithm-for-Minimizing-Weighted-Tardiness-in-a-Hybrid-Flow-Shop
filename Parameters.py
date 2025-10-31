import pandas as pd
import numpy as np
import random
import math
import time
from datetime import timedelta
from copy import deepcopy
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

#=== Settings ===#

# path to data
path_jobs = r'C:.xlsx'

# problem parameters
list_workcenters = ['ASSEMBLY','SMT', 'PLASTIC', 'PTH']
dict_machines = {'ASSEMBLY':['L1','L2','L3','L4','L5','L6','L7','L8'],
            'PTH':['PTH 1','PTH 2','PTH 3'],
            'SMT':['SMT 1','SMT 2','SMT 3'],
            'PLASTIC':['INJ 04','INJ 1','INJ 5',
                        'INJ 10','INJ 8','INJ 7',
                        'INJ 6','INJ 3','INJ 2',
                        'INJ 9','INJ 11']}

buffer_time = 5 # in days

# shifts per machine
dict_machine_turns = {'L1': 3, 'L2': 3, 'L3': 3, 'L4': 2,'L5':2, 'L6': 2, 'L7': 3, 'L8': 3,
                      'PTH 1': 3,'PTH 2': 3,'PTH 3': 3,
                      'SMT 1':3,'SMT 2':3,'SMT 3': 3,
                      'INJ 04': 3,'INJ 1': 3,'INJ 5': 3,'INJ 10': 3,'INJ 8': 3,'INJ 7': 3,'INJ 6': 3,
                      'INJ 3': 3,'INJ 2': 3,'INJ 9': 3,'INJ 11': 3}

# hours per shift
time_for_turn = 7.5

# schedule start date
schedule_date = pd.to_datetime('2025-03-01')

# Data import
raw_data_df = pd.read_excel(path_jobs, sheet_name=1 )

print(raw_data_df)

def generate_taguchi_matrix():
    # Defining parameters and levels as requested
    params = {
        'popsize': [50, 100, 150],
        'selection': ['tournament', 'roulette', 'rank'],
        'crossover': ['OX', 'PMX', 'CX'],
        'pmut': [0.01, 0.02],
        'replacement': ['elitism', 'hill_climbing', 'SA'],
        'MaxGen': [150, 300, 500],
        'restart': [True, False]
    }

    # Taguchi Matrix L18 (2^1 × 3^6) - 18 experiments
    # Each row represents a unique parameter combination
    taguchi_matrix = [
        # Experiment 1-9: restart=True
        {'restart': True, 'popsize': 50, 'selection': 'tournament', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'simple', 'MaxGen': 150},
        #{'restart': True, 'popsize': 50, 'selection': 'roulette', 'crossover': 'PMX', 'pmut': 0.02, 'replacement': 'hill_climbing', 'MaxGen': 300},
        {'restart': True, 'popsize': 50, 'selection': 'rank', 'crossover': 'CX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 500},
        {'restart': True, 'popsize': 100, 'selection': 'tournament', 'crossover': 'OX', 'pmut': 0.02, 'replacement': 'hill_climbing', 'MaxGen': 500},
        #{'restart': True, 'popsize': 100, 'selection': 'roulette', 'crossover': 'PMX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 150},
        {'restart': True, 'popsize': 100, 'selection': 'rank', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'simple', 'MaxGen': 300},
        #{'restart': True, 'popsize': 150, 'selection': 'tournament', 'crossover': 'PMX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 500},
        {'restart': True, 'popsize': 150, 'selection': 'roulette', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'simple', 'MaxGen': 150},
        {'restart': True, 'popsize': 150, 'selection': 'rank', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 300},
        
        # Experiment 10-18: restart=False
        {'restart': False, 'popsize': 50, 'selection': 'tournament', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'simple', 'MaxGen': 150},
        #{'restart': True, 'popsize': 50, 'selection': 'roulette', 'crossover': 'PMX', 'pmut': 0.02, 'replacement': 'hill_climbing', 'MaxGen': 300},
        {'restart': False, 'popsize': 50, 'selection': 'rank', 'crossover': 'CX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 500},
        {'restart': False, 'popsize': 100, 'selection': 'tournament', 'crossover': 'OX', 'pmut': 0.02, 'replacement': 'hill_climbing', 'MaxGen': 500},
        #{'restart': True, 'popsize': 100, 'selection': 'roulette', 'crossover': 'PMX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 150},
        {'restart': False, 'popsize': 100, 'selection': 'rank', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'simple', 'MaxGen': 300},
        #{'restart': True, 'popsize': 150, 'selection': 'tournament', 'crossover': 'PMX', 'pmut': 0.01, 'replacement': 'SA', 'MaxGen': 500},
        {'restart': False, 'popsize': 150, 'selection': 'roulette', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'simple', 'MaxGen': 150},
        {'restart': False, 'popsize': 150, 'selection': 'rank', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 300},
        ]
    return taguchi_matrix

taguchi_matrix_2 = [
    # Experiment 1-9: restart=True
    {'restart': True, 'popsize': 150, 'selection': 'rank', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 300},
    {'restart': True, 'popsize': 50, 'selection': 'roulette', 'crossover': 'OX', 'pmut': 0.02, 'replacement': 'SA', 'MaxGen': 300},
    {'restart': True, 'popsize': 100, 'selection': 'roulette', 'crossover': 'CX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 500},
    {'restart': True, 'popsize': 150, 'selection': 'tournament', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'SA', 'MaxGen': 300},

    {'restart': False, 'popsize': 150, 'selection': 'rank', 'crossover': 'OX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 300},
    {'restart': False, 'popsize': 50, 'selection': 'roulette', 'crossover': 'OX', 'pmut': 0.02, 'replacement': 'SA', 'MaxGen': 300},
    {'restart': False, 'popsize': 100, 'selection': 'roulette', 'crossover': 'CX', 'pmut': 0.01, 'replacement': 'hill_climbing', 'MaxGen': 500},
    {'restart': False, 'popsize': 150, 'selection': 'tournament', 'crossover': 'CX', 'pmut': 0.02, 'replacement': 'SA', 'MaxGen': 300}
    
    ]