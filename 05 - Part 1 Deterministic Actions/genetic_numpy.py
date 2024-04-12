import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


class GA_discrete_actions:

    def __init__(self, plant_params, spot, utc_time, actions_space):

        self.plant_params = plant_params
        self.spot = spot
        self.utc_time = utc_time
        self.individual_size = len(spot)
        self.action_space = actions_space

    def train(self, config, total_generations, tune_mode: bool):
        
        # Initialise random population
        self.population = np.random.choice(
            a=self.action_space, size=(config["POP_SIZE"], self.individual_size)
        )

        # Initialise elite size according to the hyperparameters
        self.elite_size = int(config["ELITISM"] * config["POP_SIZE"])

        # Initialise mutation rate
        self.mutation_rate = config["INITIAL_MUTATION_RATE"]

        # Initialise generation counter
        generation = 0

        if not tune_mode:
            # Initialise lists for storing intermediate values
            best_fitnesses = []
            avg_fitnesses = []
            worst_fitnesses = []

        # Launch the actual evolution loop over total_generations
        for generation in (pbar := tqdm(range(total_generations), disable=tune_mode)):
            
            # Evaluate the fitness of the population
            fitnesses = evaluate_fitness(self.population, self.plant_params, self.spot)

            # Sort population ascendingly
            fitness_indices = fitnesses.argsort()
            sorted_fitnesses = fitnesses[fitness_indices]
            sorted_population = self.population[fitness_indices]

            # Identify the elite for reproduction
            elite = sorted_population[-self.elite_size :]
            sorted_elite_fitnesses = sorted_fitnesses[-self.elite_size :]

            min_fitness = sorted_elite_fitnesses.min()
            max_fitness = sorted_elite_fitnesses.max()

            # If all fitnesses are equal, set weighting to uniform distribution
            if min_fitness == max_fitness:
                elite_weighting = np.full_like(
                    sorted_elite_fitnesses, 1 / len(sorted_elite_fitnesses), dtype=float
                )
            # If not all are equal, weight according to fitness
            else:
                elite_weighting = (sorted_elite_fitnesses - min_fitness) / (
                    max_fitness - min_fitness
                )
                elite_weighting /= elite_weighting.sum()

            # Initialise the new population
            new_population = np.zeros((config["POP_SIZE"], self.individual_size))

            # Keep previous population according to survival rate of the best individuals
            survival_cutoff = int(
                np.floor(config["SURVIVAL_RATE"] * config["POP_SIZE"])
            )
            new_population[0:survival_cutoff] = sorted_population[-survival_cutoff:]

            # Crossover the elite only, rest of population doesn't cross over
            # Produce as many children as needed to fill the population
            for child_id in np.arange(survival_cutoff + 1, config["POP_SIZE"]):
                # Select two random indices using roulette wheel selection
                i0 = np.random.choice(
                    a=self.elite_size, p=elite_weighting, replace=True
                )
                i1 = np.random.choice(
                    a=self.elite_size, p=elite_weighting, replace=True
                )

                # Randomly select parts of parent 1 and 2 and add to new population
                child = np.copy(elite[i0])
                dna2_indices = np.random.randint(2, size=child.size)
                indices = np.where(dna2_indices)
                child[indices] = elite[i1][indices]
                new_population[child_id] = child

            # Mutate the new population
            for dna_id in range(config["POP_SIZE"]):
                if np.random.random_sample() < self.mutation_rate:

                    # Randomly select which element to mutate and insert random integer parameters
                    mutation_values = np.random.choice(a=self.action_space, size=1)
                    positions = np.random.choice(a=np.arange(0, len(new_population[dna_id])), size=1)

                    new_population[dna_id][positions] = mutation_values


            # Overwrite the old population with the new one
            self.population = new_population

            # Append values, show progress
            if not tune_mode:
                # Append the plot information
                best_fitnesses.append(np.max(fitnesses))
                avg_fitnesses.append(np.mean(fitnesses))
                worst_fitnesses.append(np.min(fitnesses))

                # Update progress bar
                pbar.set_description(
                    f"Generation: {generation}\nBest: {best_fitnesses[-1]:.2f}\nAverage: {avg_fitnesses[-1]:.2f}",
                    refresh=True,
                )

            if tune_mode:
                start_index = int(len(sorted_fitnesses) * 0.8)
                train.report({"fitness": np.mean(sorted_fitnesses[start_index:])})

            # Increment counter
            generation += 1

            # Decrease mutation rate
            self.mutation_rate = np.where(
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

        if not tune_mode:
            history = pd.DataFrame(
                {
                    "generation": np.arange(0, len(best_fitnesses)),
                    "best": best_fitnesses,
                    "average": avg_fitnesses,
                    "worst": worst_fitnesses,
                }
            )

            return self.population, fitnesses, history


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


def evaluate_fitness(population, ps_params, prices):

    # To be written according to pumped storage optimisation problem
    fitness_scores = np.array([])

    for parameter_combination in population:

        water_level = ps_params["INITIAL_WATER_LEVEL"]
        fitness_score = 0

        for action, price in zip(parameter_combination, prices):
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
