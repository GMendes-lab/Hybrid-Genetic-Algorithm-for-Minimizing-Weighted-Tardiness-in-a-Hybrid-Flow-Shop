# Hybrid-Genetic-Algorithm-for-Minimizing-Weighted-Tardiness-in-a-Hybrid-Flow-Shop

## 📋 Project Overview

This project implements a sophisticated Genetic Algorithm (GA) for optimizing production scheduling in manufacturing environments. The system handles complex constraints including machine eligibility, setup times, due dates, and production flow dependencies across multiple workcenters.

## 🎯 Key Features

- **Multi-workcenter Scheduling**: Handles production flow across ASSEMBLY, SMT, PLASTIC, and PTH workcenters
- **Real-world Constraints**: Considers machine eligibility, setup times, processing times, and due dates
- **Advanced Genetic Operators**: Implements OX, PMX, and CX crossover with multiple mutation strategies
- **Multiple Selection Methods**: Tournament, Roulette Wheel, and Rank-based selection
- **Intelligent Replacement**: Simple, Hill Climbing, and Simulated Annealing replacement strategies
- **Taguchi Experimental Design**: Systematic parameter optimization using Taguchi methods
- **Stagnation Handling**: Population reactivation mechanisms to avoid local optima

## 🏗️ System Architecture

### Core Components

1. **Initialization Module**
   - `generate_optimized_population()`: Creates initial schedules using EDD-based allocation
   - `optimize_sequence_with_setup()`: Minimizes setup times between operations

2. **Genetic Operators**
   - **Selection**: Tournament, Roulette, Rank selection
   - **Crossover**: OX, PMX crossover methods
   - **Mutation**: Shuffle, swap, inversion, scramble mutations
   - **Replacement**: Simple, Hill Climbing, Simulated Annealing

3. **Fitness Evaluation**
   - `calculate_fitness_population()`: Computes weighted tardiness based on due dates
   - `calculate_completion_time()`: Determines operation timelines considering dependencies

4. **Constraint Handling**
   - Machine eligibility rules
   - Setup time matrices
   - Processing time calculations
   - Production flow precedence

## 📊 Input Data Structure

The system expects production data with the following columns:
- `JOB`: Job identifier
- `WORKCENTER`: Production workcenter
- `PRODUCT`: Product type
- `DUE DATE`: Job due date
- `QUANTITY`: Production quantity
- `GOAL`: Production goal rate
- `MACHINE`: Machine assignments
- `CONSTRAINTS`: Operational constraints (Mandatory/Preferential)

## ⚙️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
pandas
numpy
matplotlib
