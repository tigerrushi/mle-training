import argparse
import configparser
import pickle
from scipy.stats import randint
import sys
import os
import mlflow
import mlflow.sklearn


sys.path.insert(
    1,
    "logs",
)

import logger_01 as l

logger = l.configure_logger(
    log_file=os.path.join(
        "logs/logging_files",
        "custom_config.log",
    )
)
logger.info("Starting train.py")
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


def model_training(model_instance, housing_prepared, housing_labels, model_path):
    logger.info("Training Started")
    model_instance.fit(housing_prepared, housing_labels)
    logger.info("Training COmpleted")
    pred = model_instance.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(housing_labels, pred)
    logger.info(f"{model_instance} algorithm, mse : {mse}, rmse {rmse}, mae {mae}")
    logger.info("Saving model")
    pickle.dump(
        model_instance,
        open(model_path + "/" + str(model_instance)[:-2] + ".pkl", "wb"),
    )


def training_with_gridsearchCV(
    housing_prepared, housing_labels, model_path, strat_test_set, exp_name
):
    EXPERIMENT_NAME = exp_name
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="run 1"):

        with mlflow.start_run(
            experiment_id=EXPERIMENT_ID, run_name="linear_regression", nested=True
        ):

            lin_reg = LinearRegression()
            lin_reg.fit(housing_prepared, housing_labels)

            housing_predictions = lin_reg.predict(housing_prepared)
            lin_mse = mean_squared_error(housing_labels, housing_predictions)

            lin_rmse = np.sqrt(lin_mse)
            lin_rmse

            lin_mae = mean_absolute_error(housing_labels, housing_predictions)
            lin_mae
            mlflow.log_metrics({"MSE": lin_mse, "RMSE": lin_rmse, "MAE": lin_mae})
            mlflow.sklearn.log_model(sk_model=lin_reg, artifact_path="model")
            pickle.dump(
                lin_reg,
                open(model_path + "/linear_regresion.pkl", "wb"),
            )
        with mlflow.start_run(
            experiment_id=EXPERIMENT_ID, run_name="Regression_Tree", nested=True
        ):
            tree_reg = DecisionTreeRegressor(random_state=42)
            tree_reg.fit(housing_prepared, housing_labels)

            housing_predictions = tree_reg.predict(housing_prepared)
            tree_mse = mean_squared_error(housing_labels, housing_predictions)
            tree_rmse = np.sqrt(tree_mse)
            tree_rmse
            mlflow.log_metrics({"MSE": tree_mse, "RMSE": tree_rmse})
            mlflow.sklearn.log_model(sk_model=tree_rmse, artifact_path="model")

            pickle.dump(
                tree_reg,
                open(model_path + "/Decision_tree_reg.pkl", "wb"),
            )
        mlflow.sklearn.autolog()
        with mlflow.start_run(
            experiment_id=EXPERIMENT_ID, run_name="RandomForestTree", nested=True
        ):
            param_distribs = {
                "n_estimators": randint(low=1, high=200),
                "max_features": randint(low=1, high=8),
            }

            forest_reg = RandomForestRegressor(random_state=42)
            rnd_search = RandomizedSearchCV(
                forest_reg,
                param_distributions=param_distribs,
                n_iter=10,
                cv=2,
                scoring="neg_mean_squared_error",
                random_state=42,
            )
            rnd_search.fit(housing_prepared, housing_labels)
            cvres = rnd_search.cv_results_
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                print(np.sqrt(-mean_score), params)

        with mlflow.start_run(
            experiment_id=EXPERIMENT_ID, run_name="GridSearchCV", nested=True
        ):
            param_grid = [
                # try 12 (3×4) combinations of hyperparameters
                {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
                # then try 6 (2×3) combinations with bootstrap set as False
                {
                    "bootstrap": [False],
                    "n_estimators": [3, 10],
                    "max_features": [2, 3, 4],
                },
            ]

            forest_reg = RandomForestRegressor(random_state=42)
            # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
            grid_search = GridSearchCV(
                forest_reg,
                param_grid,
                cv=2,
                scoring="neg_mean_squared_error",
                return_train_score=True,
            )
            grid_search.fit(housing_prepared, housing_labels)

            grid_search.best_params_
            cvres = grid_search.cv_results_
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                print(np.sqrt(-mean_score), params)

            feature_importances = grid_search.best_estimator_.feature_importances_
            sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

            final_model = grid_search.best_estimator_
            print("Final Model : ", final_model)
        mlflow.sklearn.autolog(disable=True)


def run_model_training(exp_name):

    logger.info("Configparser instantiating")
    config = configparser.ConfigParser()

    path = "config.ini"
    config.read(path)
    logger.info("Initializing Default parameters")
    housing_prepared = config["ProcessedData"]["housing_prepared"]
    housing_labels = config["ProcessedData"]["housing_labels"]
    strat_test_set = config["ProcessedData"]["strat_test_set"]
    model_path = config["ProcessedData"]["model_path"]

    logger.info("Instantiating Argument Parser")

    logger.info("Reading CSV files from default parameters")
    housing_prepared = pd.read_csv(housing_prepared)
    housing_labels = pd.read_csv(housing_labels)
    strat_test_set = pd.read_csv(strat_test_set)

    logger.info("Calling Model_Training Function")
    # model_training(LinearRegression(), housing_prepared, housing_labels, model_path)

    # model_training(
    #     DecisionTreeRegressor(random_state=42),
    #     housing_prepared,
    #     housing_labels,
    #     model_path,
    # )

    # model_training(
    #     RandomForestRegressor(n_estimators=100, random_state=42),
    #     housing_prepared,
    #     housing_labels,
    #     model_path,
    # )

    training_with_gridsearchCV(
        housing_prepared, housing_labels, model_path, strat_test_set, exp_name
    )

    logger.info(
        "Execution completed\n**********************************************************"
    )


if __name__ == "__main__":

    logger.info("Configparser instantiating")
    config = configparser.ConfigParser()

    path = "config.ini"
    config.read(path)
    logger.info("Initializing Default parameters")
    housing_prepared = config["ProcessedData"]["housing_prepared"]
    housing_labels = config["ProcessedData"]["housing_labels"]
    strat_test_set = config["ProcessedData"]["strat_test_set"]
    model_path = config["ProcessedData"]["model_path"]

    logger.info("Instantiating Argument Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--housing_prepared",
        help="Enter the output folder path to store the dataset",
        default=housing_prepared,
    )
    parser.add_argument(
        "--housing_labels",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",
        default=housing_labels,
    )
    parser.add_argument(
        "--strat_test_set",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",
        default=strat_test_set,
    )
    parser.add_argument(
        "--model_path",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",
        default=model_path,
    )
    args = parser.parse_args()

    housing_prepared = args.housing_prepared
    housing_labels = args.housing_labels
    strat_test_set = args.strat_test_set
    model_path = args.model_path

    logger.info("Reading CSV files from default parameters")
    housing_prepared = pd.read_csv(housing_prepared)
    housing_labels = pd.read_csv(housing_labels)
    strat_test_set = pd.read_csv(strat_test_set)

    logger.info("Calling Model_Training Function")
    model_training(LinearRegression(), housing_prepared, housing_labels, model_path)

    model_training(
        DecisionTreeRegressor(random_state=42),
        housing_prepared,
        housing_labels,
        model_path,
    )

    model_training(
        RandomForestRegressor(n_estimators=100, random_state=42),
        housing_prepared,
        housing_labels,
        model_path,
    )

    logger.info(
        "Execution completed\n**********************************************************"
    )
