# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:16:14 2025

@author: Francisco

"""

import random

#Parameters
TARGET = "HELLO WORLD"
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
GENERATIONS = 1000

#Generates a random string of a given length
def generate_individual(length):
    return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ ") for _ in range(length))

#Calculates the fitness of an individual based on how closely it matches the target
def calculate_fitness(individual):
    fitness = 0
    for i in range(len(TARGET)):
#        if i < len(individual) and individual[i] == TARGET[i]:
        if individual[i] == TARGET[i]:
           fitness += 1
    return fitness

#Selects two parents based on their fitness (roulette wheel selection)
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:  # Handle case where all fitnesses are zero
        return random.sample(population, 2)

    rnd1 = random.uniform(0, total_fitness)
    rnd2 = random.uniform(0, total_fitness)
    
    parent1 = None
    parent2 = None
    current_sum = 0
    for i, individual in enumerate(population):
        current_sum += fitnesses[i]
        if parent1 is None and current_sum >= rnd1:
            parent1 = individual
        if parent2 is None and current_sum >= rnd2:
            parent2 = individual
        if parent1 is not None and parent2 is not None:
            break
        
    return parent1, parent2

#Performs single-point crossover between two parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

#Mutates an individual by randomly changing characters
def mutate(individual, mutation_rate):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    return "".join(mutated_individual)

def run_genetic_algorithm():
    #Generate n (POPULATION_SIZE) strings of a given lenght
    population = [generate_individual(len(TARGET)) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        fitnesses = [calculate_fitness(ind) for ind in population]

        #Check for solution
        best_fitness = max(fitnesses)
        best_individual_index = fitnesses.index(best_fitness)
        best_individual = population[best_individual_index]

        if best_individual == TARGET:
            print(f"Target found in generation {generation}: {best_individual}")
            return

        print(f"Generation {generation}: Best Fitness = {best_fitness}/{len(TARGET)}, Best Individual = {best_individual}")

        new_population = []
        for _ in range(POPULATION_SIZE // 2):  # Create pairs of children
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, MUTATION_RATE))
            new_population.append(mutate(child2, MUTATION_RATE))
        
        population = new_population

    print("Genetic algorithm finished without finding the exact target.")
    print(f"Final best individual: {best_individual}, Fitness: {best_fitness}/{len(TARGET)}")

#Run the algorithm
if __name__ == "__main__":
    run_genetic_algorithm()