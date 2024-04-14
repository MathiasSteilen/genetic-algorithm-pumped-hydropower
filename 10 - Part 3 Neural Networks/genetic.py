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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super(ANN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


def encode_chromosome(model):
    """
    Encodes the model parameters into a single chromosome, which can be used for genetic algorithms.
    """
    chromosome = torch.tensor([])

    for param in model.parameters():
        chromosome = torch.cat((chromosome, param.view(-1)))

    return chromosome.numpy()


def decode_chromosome(model, chromosome):
    """
    Decodes the chromosome into the model parameters and updates the model with the new parameters.
    """
    chromosome = torch.tensor(chromosome, dtype=torch.float32)
    model_params = list(model.parameters())
    start = 0

    for param in model_params:
        end = start + torch.numel(param)
        param.data = chromosome[start:end].view(param.size())
        start = end


def evaluate_fitness(population, ps_params, spot_prices):

    fitnesses = []

    for individual in population:

        water_level = ps_params["INITIAL_WATER_LEVEL"]
        fitness_score = 0

        for sigmoid_value, price in zip(individual, spot_prices):
            # Pump (-1)
            if sigmoid_value < 0.33:
                if (
                    water_level + ps_params["PUMP_RATE_M3H"]
                    < ps_params["MAX_STORAGE_M3"]
                ):
                    fitness_score -= ps_params["PUMP_POWER_MW"] * price
                    water_level += ps_params["PUMP_RATE_M3H"]
                else:
                    fitness_score -= 100_000
            # Turbine (1)
            if sigmoid_value > 0.66:
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

        fitnesses.append(fitness_score)

    return np.array(fitnesses)


class GA_ANN:
    def __init__(
        self,
        plant_params,
        spot,
        hidden_layers,
        hidden_size,
    ):
        self.plant_params = plant_params
        self.spot = spot
        self.input_data = torch.cat(
            [
                torch.tensor([self.plant_params["INITIAL_WATER_LEVEL"]]),
                torch.tensor(self.spot),
            ]
        )

        # Neural Network parameters
        self.input_size = len(self.input_data)
        self.output_size = len(self.spot)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        # Initialise the model
        self.model = ANN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
        )

        # Need to disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Individual size
        self.individual_size = encode_chromosome(self.model).shape[0]

    def train(self, config, tune_mode: bool):

        # Initialise random population
        self.population = np.random.normal(
            loc=0, scale=10, size=(config["POP_SIZE"], self.individual_size)
        )
        self.population = np.array(self.population, dtype=np.float32)

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

        # Launch the actual evolution loop over TOTAL_GENERATIONS
        for generation in (
            pbar := tqdm(range(config["TOTAL_GENERATIONS"]), disable=tune_mode)
        ):

            # Set the weights of the network and make predictions
            predictions = np.zeros(
                (config["POP_SIZE"], self.output_size), dtype=np.float32
            )

            for idx, individual in enumerate(self.population):

                # Set the model weights
                decode_chromosome(self.model, individual)

                # Make predictions with the configuration
                predictions[idx] = self.model(self.input_data).numpy()

            # Evaluate the fitness of the population
            fitnesses = evaluate_fitness(
                population=predictions,
                ps_params=self.plant_params,
                spot_prices=self.spot,
            )

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
                    # Add Gaussian noise to the entire chromosome
                    new_population[dna_id] += np.random.normal(loc=0, scale=config["MUTATION_SIGMA"], size=self.individual_size)

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
                generation <= int(config["INITIAL_EXPLORATION"] * config["TOTAL_GENERATIONS"]),
                config["INITIAL_MUTATION_RATE"]
                * np.exp(
                    (
                        np.log(config["FINAL_MUTATION_RATE"])
                        - np.log(config["INITIAL_MUTATION_RATE"])
                    )
                    / (config["INITIAL_EXPLORATION"] * config["TOTAL_GENERATIONS"])
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
        timeout_s,
    ):

        # Need this line for locally defined modules to work with ray
        ray.init(runtime_env={"working_dir": "."}, ignore_reinit_error=True)

        analysis = tune.run(
            tune.with_parameters(
                self.train,
                tune_mode=True,
            ),
            config=tune_config,
            metric="fitness",
            mode="max",
            local_dir="tune_results",
            name="GA",
            search_alg=OptunaSearch(),
            time_budget_s=timeout_s,
            num_samples=-1,
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        )

        return analysis
