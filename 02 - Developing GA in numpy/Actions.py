import numpy as np
import pandas as pd
import polars as pl


class GA_discrete_actions:

    def __init__(
        self,
        dna_size,
        discrete_action_space,
        elitism=0.25,
        survival_rate=0.25,
        population_size=200,
        mutation_rate=0.01,
        mutation_decay=0.999,
        mutation_limit=0.01,
    ):

        self.dna_size = dna_size
        self.discrete_action_space = discrete_action_space
        self.elitism = elitism
        self.survival_rate = survival_rate
        self.population_size = population_size
        self.elite_size = int(self.elitism * self.population_size)
        self.mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_limit = mutation_limit

        self.best_dna = None
        self.best_fitness = None
        self.average_fitness = None

        self.initial_population = self.__create_random_population()

    def evolve(self, population, fitnesses):

        assert len(population) == self.population_size
        assert len(fitnesses) == len(population)
        assert isinstance(population, np.ndarray)

        # Sort population and fitnesses
        fitness_indices = fitnesses.argsort()
        sorted_fitnesses = fitnesses[fitness_indices]
        sorted_population = population[fitness_indices]

        # Report the best fitness and parameters
        self.best_dna = sorted_population[-1]
        self.best_fitness = sorted_fitnesses[-1]
        self.average_fitness = np.mean(sorted_fitnesses)

        # Only keep the elite of the population and get their weighting
        elite = sorted_population[-self.elite_size :]
        sorted_elite_fitnesses = sorted_fitnesses[-self.elite_size :]

        # Calculate weighting for elite
        min_fitness = sorted_elite_fitnesses.min()
        max_fitness = sorted_elite_fitnesses.max()

        if min_fitness == max_fitness:
            # If all fitnesses are equal, set weighting to uniform distribution
            fitnesses_weighting = np.full_like(
                sorted_elite_fitnesses, 1 / len(sorted_elite_fitnesses), dtype=np.float32
            )
        else:
            fitnesses_weighting = (sorted_elite_fitnesses - min_fitness) / (
                max_fitness - min_fitness
            )
            fitnesses_weighting /= fitnesses_weighting.sum()

        new_population = np.zeros((self.population_size, self.dna_size))

        # Keep the population according to survival rate
        survival_cutoff = int(np.floor(self.survival_rate * self.population_size))
        new_population[0:survival_cutoff] = sorted_population[-survival_cutoff:]

        # Crossover the elite only, rest of population doesn't cross over
        for child_id in np.arange(survival_cutoff + 1, self.population_size):
            i0 = np.random.choice(
                a=self.elite_size, p=fitnesses_weighting, replace=True
            )
            i1 = np.random.choice(
                a=self.elite_size, p=fitnesses_weighting, replace=True
            )

            new_dna = self.__crossover(elite[i0], elite[i1])
            
            assert isinstance(new_dna, np.ndarray)

            new_population[child_id] = new_dna

        # Mutate the new population
        for dna_id in range(self.population_size):
            new_population[dna_id] = self.__mutate(new_population[dna_id])

        # Adjust mutation rate
        self.mutation_rate *= self.mutation_decay
        self.mutation_rate = np.maximum(self.mutation_rate, self.mutation_limit)

        assert new_population.shape == self.initial_population.shape

        return new_population

    def __create_random_population(self):
        population = np.random.choice(
            a=self.discrete_action_space, size=(self.population_size, self.dna_size)
        )
        return population

    def __mutate(self, dna):

        # If random dice roll (between zero and one) is less than mutation
        # rate (between zero and one) then inject noise into the dna.
        if np.random.random_sample() < self.mutation_rate:

            # Randomly select which element to mutate and insert random integer parameters
            mutation_values = np.random.choice(a=self.discrete_action_space, size=1)
            positions = np.random.choice(a=np.arange(0, len(dna)), size=1)

            dna[positions] = mutation_values

        return dna

    def __crossover(self, dna1, dna2):
        assert len(dna1) == len(dna2)
        # set child's DNA to be of the first parent
        child = np.copy(dna1)
        # replace random positions with dna from second parent
        dna2_indices = np.random.randint(2, size=child.size)
        indices = np.where(dna2_indices)
        child[indices] = dna2[indices]

        return child


def evaluate_fitness(population, ps_params, prices):

    # To be written according to pumped storage optimisation problem
    fitness_scores = np.array([])

    for parameter_combination in population:

        water_level = ps_params["INITIAL_WATER_LEVEL"]
        fitness_score = 0

        for action, price in zip(parameter_combination, prices["spot"]):
            # Pump (1)
            if action == 1:
                if (
                    water_level + ps_params["PUMP_RATE_M3H"]
                    < ps_params["MAX_STORAGE_M3"]
                ):
                    fitness_score -= ps_params["PUMP_POWER_MW"] * price
                    water_level += ps_params["PUMP_RATE_M3H"]
                else:
                    fitness_score -= 100_000
            # Turbine (-1)
            if action == -1:
                if (
                    water_level - ps_params["TURBINE_RATE_M3H"]
                    > ps_params["MIN_STORAGE_M3"]
                ):
                    fitness_score += ps_params["TURBINE_POWER_MW"] * price
                    water_level -= ps_params["TURBINE_RATE_M3H"]
                else:
                    fitness_score -= 100_000
            # Do nothing (0)
            # Nothing happens to the fitness score and the water level

        fitness_scores = np.append(fitness_scores, fitness_score)

    return fitness_scores