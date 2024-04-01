import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

def trainable(config):
    train.report({"metric": config["x"]**2})