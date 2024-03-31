import pandas as pd
import numpy as np
from pulp import (
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    LpContinuous,
    LpInteger,
    value,
)


class MILP:
    def __init__(self, utc_time, spot, plant_params) -> None:
        self.utc_time = utc_time
        self.spot = spot
        self.plant_params = plant_params

    def __enframe(self, model):
        best_profile = (
            pd.DataFrame(
                {
                    "variable": [v.name for v in model.variables()],
                    "value": [v.value() for v in model.variables()],
                }
            )
            .assign(
                name=lambda x: x["variable"].apply(lambda r: r.rsplit("_", 1)[0]),
                hour=lambda x: x["variable"]
                .apply(lambda r: r.rsplit("_", 1)[1])
                .astype(int),
            )
            .drop("variable", axis=1)
            .pivot(index="hour", columns="name", values="value")
            .reset_index()
            .rename_axis(None, axis=1)
            .sort_values("hour")
            .assign(
                action=lambda x: np.select(
                    condlist=[
                        x["pump_dummy"] == 1,
                        x["turbine_dummy"] == 1,
                    ],
                    choicelist=[
                        -1,
                        1,
                    ],
                    default=0,
                ),
                colour_id=lambda x: np.select(
                    condlist=[
                        x["pump_dummy"] == 1,
                        x["turbine_dummy"] == 1,
                    ],
                    choicelist=[
                        "pump",
                        "turbine",
                    ],
                    default="nothing",
                ),
                utc_time=pd.to_datetime(self.utc_time),
                spot=self.spot,
            )
            .drop(["pump_dummy", "turbine_dummy", "hour"], axis=1)
        )

        return best_profile

    def solve(self):
        # Create names for decision variables
        dec_vars = range(len(self.spot))

        # The cost/revenue associated with each decision variable is the
        # day-ahead spot price provided to the class
        spot = dict(zip(dec_vars, self.spot))

        # Decision variables
        dp = LpVariable.dicts("pump_dummy", dec_vars, 0, 1, cat="Binary")
        dt = LpVariable.dicts("turbine_dummy", dec_vars, 0, 1, cat="Binary")

        # State Variables
        water_level = LpVariable.dict(
            "water_level",
            dec_vars,
            self.plant_params["MIN_STORAGE_M3"],
            self.plant_params["MAX_STORAGE_M3"],
            cat="Continuous",
        )

        # Initialise Model
        model = LpProblem(name="pumped_storage_optimisation", sense=LpMaximize)

        # Add objective function
        model += (
            lpSum(
                [
                    dt[i] * self.plant_params["TURBINE_POWER_MW"] * self.spot[i]
                    - dp[i] * self.plant_params["PUMP_POWER_MW"] * self.spot[i]
                    for i in dec_vars
                ]
            ),
            "revenue",
        )

        # Add constraints
        for i in dec_vars:
            # Don't pump and turbine at the same time
            model += (dt[i] + dp[i] <= 1, f"no_simultaneous_pump_and_turbine_upper_{i}")
            model += (dt[i] + dp[i] >= 0, f"no_simultaneous_pump_and_turbine_lower_{i}")

            # Add constraint to update water level
            if i == 0:
                model += (
                    water_level[i]
                    == self.plant_params["INITIAL_WATER_LEVEL"]
                    + dp[i] * self.plant_params["PUMP_RATE_M3H"]
                    - dt[i] * self.plant_params["TURBINE_RATE_M3H"],
                    f"initial_water_level_{i}",
                )
            else:
                # Update water level based on previous level, pumping, and turbine action
                model += (
                    water_level[i]
                    == water_level[i - 1]
                    + dp[i] * self.plant_params["PUMP_RATE_M3H"]
                    - dt[i] * self.plant_params["TURBINE_RATE_M3H"],
                    f"water_level_update_{i}",
                )

            # Add constraints for water level boundaries
            model += (
                water_level[i] <= self.plant_params["MAX_STORAGE_M3"],
                f"max_storage_{i}",
            )
            model += (
                water_level[i] >= self.plant_params["MIN_STORAGE_M3"],
                f"min_storage_{i}",
            )

        # Solve the model
        model.solve()

        best_profile = self.__enframe(model)

        return model, LpStatus[model.status], best_profile
