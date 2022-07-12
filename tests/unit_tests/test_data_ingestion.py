import sys
import pandas as pd

import pytest

sys.path.insert(1, "../../src/")
from data_ingest import ingest_data


@pytest.fixture
def housing():
    return pd.read_csv(
        r"C:\Users\rushikesh.naik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\rushikesh\assignment__1_2\mle-training\data\raw\housing\housing\housing.csv"
    )


@pytest.fixture
def processed_datapath():
    return r"C:\Users\rushikesh.naik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\rushikesh\assignment__1_2\mle-training\data\processed\housing_prepared.csv"


def test_shape_loading_housing_data():

    df = ingest_data.load_housing_data(
        "C:/Users/rushikesh.naik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/rushikesh/assignment__1_2/mle-training/data/raw/housing/housing"
    )
    assert (20640, 10) == df.shape


# test_type_loading_housing_data()


def test_data_preprocessing(housing):
    assert data_preprocessing(housing, processed_datapath)


def test_sdf():
    pass
