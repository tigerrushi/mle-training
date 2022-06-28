import sys
import pandas as pd

sys.path.insert(1, "../../src/")
from data_ingest import ingest_data


def test_type_loading_housing_data():

    df = ingest_data.load_housing_data("C:/Users/rushikesh.naik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/rushikesh/assignment__1_2/mle-training/data/raw/housing/housing")
    assert (20640, 10) == df.shape


#test_type_loading_housing_data()


