import argparse, configparser
import os
import tarfile
import pickle

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer


def fetch_housing_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = housing_path+"/housing.csv"
    print(csv_path)
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def data_preprocessing(housing, processed_datapath):
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    housing_prepared.to_csv(
        os.path.join(processed_datapath, "housing_prepared.csv"), index=False
    )
    housing_labels.to_csv(
        os.path.join(processed_datapath, "housing_labels.csv"), index=False
    )
    strat_test_set.to_csv(
        os.path.join(processed_datapath, "strat_test_set.csv"), index=False
    )
    pickle.dump(
            imputer,
            open(processed_datapath + "/imputer.pkl", "wb"),
    )

if __name__ == "__main__":

    config = configparser.ConfigParser()
    path = 'C:/Users/rushikesh.naik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/rushikesh/assignment__1_2/mle-training/config.ini'
    config.read(path)

    output_conf_path = config['Address']['output_path']
    processed_conf_path = config['Address']['processed_datapath']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", help="Enter the output folder path to store the dataset", default=output_conf_path
    )
    parser.add_argument(
        "--processed_datapath",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",default=processed_conf_path

    )
    args = parser.parse_args()

    output_path = args.output_path
    processed_datapath = args.processed_datapath

    print(output_path, processed_datapath)

    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_PATH = os.path.join(output_path, "housing")
    HOUSING_URL = DOWNLOAD_ROOT + f"datasets/housing/housing.tgz"

    # calling the data download function
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing_data = load_housing_data(housing_path=HOUSING_PATH)
    print(HOUSING_PATH)

    data_preprocessing(
        housing=housing_data, processed_datapath=processed_datapath
    )
