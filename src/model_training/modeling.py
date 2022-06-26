import argparse
import configparser
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeRegressor


def model_training(
    model_instance, housing_prepared, housing_labels, model_path
):
    model_instance.fit(housing_prepared, housing_labels)

    pred = model_instance.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(housing_labels, pred)
    print(f"{model_instance} algorithm, mse : {mse}, rmse {rmse}, mae {mae}")
    pickle.dump(
        model_instance,
        open(model_path + "/" + str(model_instance)[:-2] + ".pkl", "wb"),
    )


if __name__ == "__main__":
    config = configparser.ConfigParser()
    path = "C:/Users/rushikesh.naik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/rushikesh/assignment__1_2/mle-training/config.ini"
    config.read(path)

    housing_prepared = config["ProcessedData"]["housing_prepared"]
    housing_labels = config["ProcessedData"]["housing_labels"]
    strat_test_set = config["ProcessedData"]["strat_test_set"]
    model_path = config["ProcessedData"]["model_path"]

    housing_prepared = pd.read_csv(housing_prepared)
    housing_labels = pd.read_csv(housing_labels)
    strat_test_set = pd.read_csv(strat_test_set)

    model_training(
        LinearRegression(), housing_prepared, housing_labels, model_path
    )

    model_training(
        DecisionTreeRegressor(random_state=42), housing_prepared, housing_labels, model_path
    )

    model_training(
        RandomForestRegressor(n_estimators=100, random_state=42), housing_prepared, housing_labels, model_path
    )


