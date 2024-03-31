import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
from tqdm.notebook import tqdm
import os
from deap import base, creator, tools, algorithms

class GA_Actions:
    def __init__(self, spot, utc_time, plant_params):
        self.spot = spot
        self.utc_time = utc_time
        self.plant_params = plant_params

    def train_one_generation(self):
        
        
        pass

    def tune(self):
        pass
