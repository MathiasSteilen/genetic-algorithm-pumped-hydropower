import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
from tqdm.notebook import tqdm
import os
from deap import base, creator, tools, algorithms
from objproxies import CallbackProxy
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
import copy

def evaluate_fitness(individual, ps_params, spot_prices):

    water_level = ps_params["INITIAL_WATER_LEVEL"]
    fitness_score = 0

    for action, price in zip(individual, spot_prices):
        # Pump (-1)
        if action == -1:
            if (
                water_level + ps_params["PUMP_RATE_M3H"]
                < ps_params["MAX_STORAGE_M3"]
            ):
                fitness_score -= ps_params["PUMP_POWER_MW"] * price
                water_level += ps_params["PUMP_RATE_M3H"]
            else:
                fitness_score -= 100_000
        # Turbine (1)
        if action == 1:
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

    return fitness_score


class GA_Actions_Tournament:
    def __init__(self, plant_params, spot, utc_time):
        self.plant_params = plant_params
        self.spot = spot
        self.utc_time = utc_time
        self.individual_size = len(spot)

    def train(
        self,
        config,
        total_generations,
        tune_mode: bool,
    ):
        # Initialise generation counter
        generation = 0

        # Objective Direction
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Invididual Structure
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialise the toolbox
        toolbox = base.Toolbox()

        # Attribute generator for individual genes
        toolbox.register("attr_action", np.random.choice, [-1, 0, 1])

        # Structure initializers
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_action,
            self.individual_size,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic Operators
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=-1,
            up=1,
            # Non linear mutation rate decay
            indpb=CallbackProxy(
                lambda: np.where(
                    generation <= int(config["INITIAL_EXPLORATION"] * total_generations),
                    config["INITIAL_MUTATION_RATE"]
                    * np.exp(
                        (
                            np.log(config["FINAL_MUTATION_RATE"])
                            - np.log(config["INITIAL_MUTATION_RATE"])
                        )
                        / (config["INITIAL_EXPLORATION"] * total_generations)
                    )
                    ** generation,
                    config["FINAL_MUTATION_RATE"],
                )
            ),
        )
        # CallbackProxy(
        #     lambda: (total_generations - generation)
        #     / total_generations
        #     * config["MUT_IND_PB"]
        # ),

        toolbox.register(
            "select", tools.selTournament, tournsize=config["TOURNAMENT_SIZE"]
        )

        # Initialise the population
        population = toolbox.population(n=config["POP_SIZE"])

        if not tune_mode:
            # Initialise lists for storing intermediate values
            best_fitnesses = []
            avg_fitnesses = []
            worst_fitnesses = []
            best_ind = None
            best_fitness = None

        # Launch the actual evolution loop over total_generations
        for generation in (pbar := tqdm(range(total_generations), disable=tune_mode)):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < config["CXPB"]:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < config["MUTPB"]:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            # Those are the ones that have been deleted
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Overwrite the current population with the offspring
            population[:] = offspring

            # Get fitnesses
            fitnesses = [ind.fitness.values[0] for ind in population]

            if not tune_mode:
                # Append the plot information
                best_fitnesses.append(np.max(fitnesses))
                avg_fitnesses.append(np.mean(fitnesses))
                worst_fitnesses.append(np.min(fitnesses))

                # Update progress bar
                pbar.set_description(
                    f"Generation: {generation}\nBest: {best_fitnesses[-1]}\nAverage: {avg_fitnesses[-1]}",
                    refresh=True,
                )

                # Get the best individual
                if best_fitness is None or np.max(fitnesses) > best_fitness:
                    best_fitness = np.max(fitnesses)
                    best_ind = population[fitnesses.index(np.max(fitnesses))]

            if tune_mode:
                # Report average fitness to Ray Tune
                train.report({"fitness": np.mean(fitnesses)})

            # Increment counter
            generation += 1

        if not tune_mode:
            history = pd.DataFrame(
                {
                    "generation": np.arange(0, len(best_fitnesses)),
                    "best": best_fitnesses,
                    "average": avg_fitnesses,
                    "worst": worst_fitnesses,
                }
            )
            return population, fitnesses, best_ind, history

    def tune(
        self,
        tune_config,
        total_generations,
        timeout_s,
    ):

        # Need this line for locally defined modules to work with ray
        ray.init(runtime_env={"working_dir": "."}, ignore_reinit_error=True)

        analysis = tune.run(
            tune.with_parameters(
                self.train,
                total_generations=total_generations,
                tune_mode=True,
            ),
            config=tune_config,
            metric="fitness",
            mode="max",
            local_dir="tune_results",
            name="GA",
            search_alg=OptunaSearch(),
            # scheduler=ASHAScheduler(
            #     time_attr="training_iteration",
            #     grace_period=int(total_generations / 2),
            #     reduction_factor=1.5,
            # ),
            time_budget_s=timeout_s,
            num_samples=-1,
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        )

        return analysis

    def _evaluate_fitness(self, individual):
        fitness = evaluate_fitness(individual, self.plant_params, self.spot)
        return (fitness,)


class GA_Actions_Elite:
    def __init__(self, plant_params, spot, utc_time):
        self.plant_params = plant_params
        self.spot = spot
        self.utc_time = utc_time
        self.individual_size = len(spot)

    def train(
        self,
        config,
        total_generations,
        tune_mode: bool,
    ):
        # Initialise generation counter
        generation = 0

        # Objective Direction
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Invididual Structure
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialise the toolbox
        toolbox = base.Toolbox()

        # Attribute generator for individual genes
        toolbox.register("attr_action", np.random.choice, [-1, 0, 1])

        # Structure initializers
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_action,
            self.individual_size,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic Operators
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=-1,
            up=1,
            # Non linear mutation rate decay
            indpb=CallbackProxy(
                lambda: np.where(
                    generation <= int(config["INITIAL_EXPLORATION"] * total_generations),
                    config["INITIAL_MUTATION_RATE"]
                    * np.exp(
                        (
                            np.log(config["FINAL_MUTATION_RATE"])
                            - np.log(config["INITIAL_MUTATION_RATE"])
                        )
                        / (config["INITIAL_EXPLORATION"] * total_generations)
                    )
                    ** generation,
                    config["FINAL_MUTATION_RATE"],
                )
            ),
        )
        # CallbackProxy(
        #     lambda: (total_generations - generation)
        #     / total_generations
        #     * config["MUT_IND_PB"]
        # ),

        toolbox.register("select", tools.selBest, fit_attr="fitness")

        # Initialise the population
        population = toolbox.population(n=config["POP_SIZE"])

        if not tune_mode:
            # Initialise lists for storing intermediate values
            best_fitnesses = []
            avg_fitnesses = []
            worst_fitnesses = []
            best_ind = None
            best_fitness = None

        # Launch the actual evolution loop over total_generations
        for generation in (pbar := tqdm(range(total_generations), disable=tune_mode)):

            # Select the next generation individuals
            selected = toolbox.select(
                population, int(config["ELITISM"] * len(population))
            )
            elite_fitnesses = np.array(
                [fit[0] for fit in map(toolbox.evaluate, selected)]
            )

            # Get weighting for the elite
            min_fitness = np.min(elite_fitnesses)
            max_fitness = np.max(elite_fitnesses)

            if int(np.round(min_fitness)) == int(np.round(max_fitness)):
                # If all fitnesses are equal, set weighting to uniform distribution
                fitnesses_weighting = np.full_like(
                    np.arange(0, len(selected)), 1 / len(selected), dtype=float
                )
                fitnesses_weighting /= fitnesses_weighting.sum()
            else:
                fitnesses_weighting = (elite_fitnesses - min_fitness) / (
                    max_fitness - min_fitness
                )
                fitnesses_weighting /= fitnesses_weighting.sum()

            # print(f"Std dev weighting: {np.std(fitnesses_weighting)}")
            # print(f"Weighting list length: {len(fitnesses_weighting)}")

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover and mutation on the offspring
            while len(offspring) < len(population):
                i1 = np.random.choice(
                    a=len(selected), replace=True, p=fitnesses_weighting
                )
                i2 = np.random.choice(
                    a=len(selected), replace=True, p=fitnesses_weighting
                )

                parent1_copy = copy.deepcopy(selected[i1])
                parent2_copy = copy.deepcopy(selected[i2])

                child1, _ = tools.cxUniform(parent1_copy, parent2_copy, indpb=0.5)

                offspring.append(child1)

            for mutant in offspring:
                if np.random.random() < config["MUTPB"]:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals again
            fitnesses = map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            # Overwrite the current population with the offspring
            population[:] = offspring

            # Get fitnesses
            fitnesses = [ind.fitness.values[0] for ind in population]

            if not tune_mode:
                # Append the plot information
                best_fitnesses.append(np.max(fitnesses))
                avg_fitnesses.append(np.mean(fitnesses))
                worst_fitnesses.append(np.min(fitnesses))

                # Update progress bar
                pbar.set_description(
                    f"Generation: {generation}\nBest: {best_fitnesses[-1]}\nAverage: {avg_fitnesses[-1]}",
                    refresh=True,
                )

                # Get the best individual
                if best_fitness is None or np.max(fitnesses) > best_fitness:
                    best_fitness = np.max(fitnesses)
                    best_ind = population[fitnesses.index(np.max(fitnesses))]

            if tune_mode:
                # Report average fitness to Ray Tune
                train.report({"fitness": np.mean(fitnesses)})

            # Increment counter
            generation += 1

        if not tune_mode:
            history = pd.DataFrame(
                {
                    "generation": np.arange(0, len(best_fitnesses)),
                    "best": best_fitnesses,
                    "average": avg_fitnesses,
                    "worst": worst_fitnesses,
                }
            )
            return population, fitnesses, best_ind, history

    def tune(
        self,
        tune_config,
        total_generations,
        timeout_s,
    ):

        # Need this line for locally defined modules to work with ray
        ray.init(runtime_env={"working_dir": "."}, ignore_reinit_error=True)

        analysis = tune.run(
            tune.with_parameters(
                self.train,
                total_generations=total_generations,
                tune_mode=True,
            ),
            config=tune_config,
            metric="fitness",
            mode="max",
            local_dir="tune_results",
            name="GA",
            search_alg=OptunaSearch(),
            # scheduler=ASHAScheduler(
            #     time_attr="training_iteration",
            #     grace_period=int(total_generations / 2),
            #     reduction_factor=1.5,
            # ),
            time_budget_s=timeout_s,
            num_samples=-1,
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        )

        return analysis

    def _evaluate_fitness(self, individual):
        fitness = evaluate_fitness(individual, self.plant_params, self.spot)
        return (fitness,)


# if __name__ == "__main__":
#     df = pd.read_csv(r"C:\Users\mathi\OneDrive\Universitaet\2nd Semester\5_Supervised Research\genetic-algorithm-pumped-hydropower\01 - Data\example_week.csv")

#     plant_params = {
#         "EFFICIENCY": 0.75,
#         "MAX_STORAGE_M3": 5000,
#         "MIN_STORAGE_M3": 0,
#         "TURBINE_POWER_MW": 100,
#         "PUMP_POWER_MW": 100,
#         "TURBINE_RATE_M3H": 500,
#         "MIN_STORAGE_MWH": 0,
#         "INITIAL_WATER_LEVEL_PCT": 0,
#     }
#     plant_params["INITIAL_WATER_LEVEL"] = (
#         plant_params["INITIAL_WATER_LEVEL_PCT"] * plant_params["MAX_STORAGE_M3"]
#     )
#     plant_params["PUMP_RATE_M3H"] = (
#         plant_params["TURBINE_RATE_M3H"] * plant_params["EFFICIENCY"]
#     )
#     plant_params["MAX_STORAGE_MWH"] = (
#         plant_params["MAX_STORAGE_M3"] / plant_params["TURBINE_RATE_M3H"]
#     ) * plant_params["TURBINE_POWER_MW"]

#     ga_actions_solver = GA_Actions(spot=df["spot"], utc_time=df["utc_time"], plant_params=plant_params)
#     analysis = ga_actions_solver.tune(total_generations=100, timeout_s=60)
#     print(analysis.best_config)
