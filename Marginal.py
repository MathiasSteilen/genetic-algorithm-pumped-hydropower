import numpy as np
import pandas as pd
import polars as pl


class GA_discrete_actions:

    def __init__(
        self,
        dna_size,
        initial_marginal_prices,
        elitism=0.25,
        survival_rate=0.25,
        population_size=200,
        mutation_rate=0.01,
        mutation_decay=0.999,
        mutation_limit=0.01,
        mutation_size=0.5,
    ):

        self.dna_size = dna_size
        self.initial_marginal_prices = initial_marginal_prices
        self.elitism = elitism
        self.survival_rate = survival_rate
        self.population_size = population_size
        self.elite_size = int(self.elitism * self.population_size)
        self.mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_limit = mutation_limit
        self.mutation_size = mutation_size

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
                sorted_elite_fitnesses, 1 / len(sorted_elite_fitnesses)
            )
        else:
            fitnesses_weighting = (sorted_elite_fitnesses - min_fitness) / (
                max_fitness - min_fitness
            )
            fitnesses_weighting /= fitnesses_weighting.sum()

        new_population = np.zeros((self.population_size, self.dna_size))

        # Keep the elite according to survival rate
        survival_cutoff = int(np.floor(self.survival_rate * self.population_size))
        new_population[0:survival_cutoff] = sorted_population[-survival_cutoff:]

        # Crossover the rest by randomly selecting two parents from the elite
        # according to a probability distribution based on their fitness
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
            if np.random.random_sample() < self.mutation_rate:
                # Randomly add small values onto the DNA
                new_population[dna_id] = new_population[dna_id] + np.random.choice(
                    [-self.mutation_size, self.mutation_size], size=2
                )

        # Adjust mutation rate
        self.mutation_rate *= self.mutation_decay
        self.mutation_rate = np.maximum(self.mutation_rate, self.mutation_limit)

        assert new_population.shape == self.initial_population.shape

        return new_population

    def __create_random_population(self):
        population = np.column_stack(
            [
                np.random.uniform(size=self.population_size, low=0, high=200),
                np.random.uniform(size=self.population_size, low=0, high=200),
            ]
        )
        return population

    def __crossover(self, dna1, dna2):
        assert len(dna1) == len(dna2)

        # Child is half of dna1 and half of dna2
        # child = np.array([dna1[0], dna2[1]])
        child = np.mean([dna1, dna2], axis=0)

        return child
