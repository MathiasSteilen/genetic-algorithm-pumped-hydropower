import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
from tqdm.notebook import tqdm
import os
from deap import base, creator, tools, algorithms
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path

class GA_Actions:
    def __init__(self, spot, utc_time, plant_params):
        self.spot = spot
        self.utc_time = utc_time
        self.plant_params = plant_params
        self.individual_size = len(spot)

    def train(self, config, total_generations, tune_mode=False):
        if tune_mode:
            # Ray working directory
            # Write to the Tune trial directory, not the shared working dir
            tune_trial_dir = Path(train.get_context().get_trial_dir())
            with open(tune_trial_dir / "write.txt", "w") as f:
                f.write("trial saved artifact")

        TOTAL_GENERATIONS = total_generations
        INDIVIDUAL_SIZE = self.individual_size
                
        # Objective Direction
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Invidiaul Structure
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
            INDIVIDUAL_SIZE,
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
            indpb=config["MUT_IND_PB"],
        )

        toolbox.register("select", tools.selTournament, tournsize=config["TOURNAMENT_SIZE"])

        # Creating the initial
        population = toolbox.population(n=config["POP_SIZE"])

        # Begin the evolution
        best_fitnesses = []
        avg_fitnesses = []
        worst_fitnesses = []
        best_ind = None
        best_fitness = None

        for generation in tqdm(range(TOTAL_GENERATIONS)):
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

            # Increment counter
            generation += 1

            # Get fitnesses
            fitnesses = [ind.fitness.values[0] for ind in population]
            
            # Append the plot information
            best_fitnesses.append(np.max(fitnesses))
            avg_fitnesses.append(np.mean(fitnesses))
            worst_fitnesses.append(np.min(fitnesses))

            # Get the best individual
            if best_fitness is None or np.max(fitnesses) > best_fitness:
                best_fitness = np.max(fitnesses)
                best_ind = population[fitnesses.index(np.max(fitnesses))]
            
            # If tune mode is activated, don't return anything, only report
            # the performance to ray tune
            if tune_mode:
                # Report average fitness to Ray Tune
                train.report({"avg_fitness": np.mean(fitnesses)})
        
        # if tune mode is not active, return the desired values
        if not tune_mode:
            history = pd.DataFrame({
                "generation": np.arange(0, len(best_fitnesses)),
                "best": best_fitnesses,
                "average": avg_fitnesses,
                "worst": worst_fitnesses,
            })

            return best_ind, best_fitness, history
        
    # def tune(self, total_generations, timeout_s):
    #     TOTAL_GENERATIONS = total_generations
    #     TIMEOUT_S = timeout_s
    #     os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    #     analysis = tune.run(
    #         tune.with_parameters(
    #             self.train, total_generations=TOTAL_GENERATIONS, tune_mode=True
    #         ),
    #         config={
    #             "CXPB": tune.uniform(0.2, 0.8),
    #             "MUTPB": tune.uniform(0.05, 0.95),
    #             "MUT_IND_PB": tune.uniform(0.05, 0.95),
    #             "TOURNAMENT_SIZE": tune.randint(1, 10),
    #             "POP_SIZE": tune.choice([50, 250, 500, 1000, 5000]),
    #         },
    #         metric="avg_fitness",
    #         mode="max",
    #         local_dir="tune_results",
    #         name="GA",
    #         search_alg=OptunaSearch(),
    #         scheduler=ASHAScheduler(
    #             time_attr="training_iteration",
    #             grace_period=TOTAL_GENERATIONS / 2,
    #             reduction_factor=1.5,
    #         ),
    #         time_budget_s=TIMEOUT_S,
    #         num_samples=10_000,
    #         trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
    #     )

    #     return analysis

    def _evaluate_fitness(self, individual):

        # Calculate revenues from actions
        revenues = np.select(
            condlist=[
                np.array(individual) == -1,
                np.array(individual) == 1,
            ],
            choicelist=[
                -self.plant_params["PUMP_POWER_MW"] * self.spot,
                self.plant_params["TURBINE_POWER_MW"] * self.spot,
            ],
            default=0,
        )

        # Calculate water level exceedances
        water_levels = np.select(
            condlist=[
                np.array(individual) == -1,
                np.array(individual) == 1,
            ],
            choicelist=[
                self.plant_params["PUMP_RATE_M3H"],
                -self.plant_params["TURBINE_RATE_M3H"],
            ],
            default=0,
        ).cumsum()

        exceedances = (
            (water_levels >= self.plant_params["MAX_STORAGE_M3"])
            | (water_levels <= self.plant_params["MIN_STORAGE_M3"])
        ).sum()

        if exceedances > 0:
            return (revenues.sum() - 1e7,)
        else:
            return (revenues.sum(),)
        

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\mathi\OneDrive\Universitaet\2nd Semester\5_Supervised Research\genetic-algorithm-pumped-hydropower\01 - Data\example_week.csv")

    plant_params = {
        "EFFICIENCY": 0.75,
        "MAX_STORAGE_M3": 5000,
        "MIN_STORAGE_M3": 0,
        "TURBINE_POWER_MW": 100,
        "PUMP_POWER_MW": 100,
        "TURBINE_RATE_M3H": 500,
        "MIN_STORAGE_MWH": 0,
        "INITIAL_WATER_LEVEL_PCT": 0,
    }
    plant_params["INITIAL_WATER_LEVEL"] = (
        plant_params["INITIAL_WATER_LEVEL_PCT"] * plant_params["MAX_STORAGE_M3"]
    )
    plant_params["PUMP_RATE_M3H"] = (
        plant_params["TURBINE_RATE_M3H"] * plant_params["EFFICIENCY"]
    )
    plant_params["MAX_STORAGE_MWH"] = (
        plant_params["MAX_STORAGE_M3"] / plant_params["TURBINE_RATE_M3H"]
    ) * plant_params["TURBINE_POWER_MW"]

    ga_actions_solver = GA_Actions(spot=df["spot"], utc_time=df["utc_time"], plant_params=plant_params)
    analysis = ga_actions_solver.tune(total_generations=100, timeout_s=60)
    print(analysis.best_config)