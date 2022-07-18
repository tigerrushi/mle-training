"""
Package your utilities into a python package and store tha package in a private
python repo, then have docker image that contains your project install these utilities from your repo
that way you can easily control versioning of theses utilities across all project that have
a dependency on them

"""
from src.data_ingest import ingest_data
import pandas as pd
import pickle

from src.model_scoring import score
from src.model_training import train
import os
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(".", "housing")
HOUSING_URL = DOWNLOAD_ROOT + f"datasets/housing/housing.tgz"

ingest_data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
housing_path = "./housing"
df = ingest_data.load_housing_data(housing_path)
print(df)
ingest_data.data_preprocessing(df,'./processed' )
housing = pd.read_csv('./processed/housing_prepared.csv')
labels = pd.read_csv('./processed/housing_labels.csv')
model_path = './saved_models'
exp_name = 'Dude_the_great'
train.training_with_gridsearchCV(housing,
                                labels,
                                model_path,
                                'dummy',
                                exp_name = exp_name)

imputer = pickle.load(open("./processed/imputer.pkl", "rb"))
model = pickle.load(open("./saved_models/linear_regresion.pkl", "rb"))
strat_test_set = pd.read_csv('./processed/strat_test_set.csv')
score.testing_using_strat_test_set(model,
                                    imputer,
                                    strat_test_set)