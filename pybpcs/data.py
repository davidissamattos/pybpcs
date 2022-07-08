import pandas as pd
from importlib import resources


def tennis_agresti():
    with resources.path("pybpcs.csv", "tennis_agresti.csv") as df:
        return pd.read_csv(df)