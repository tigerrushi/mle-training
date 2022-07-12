import argparse

from src.data_ingest import ingest_data
from src.model_scoring import score
from src.model_training import train


def run_all(exp_name):

    ingest_data.run_data_ingest()
    train.run_model_training(exp_name)
    score.run_soring()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", help="Give Unique experiment title", default="Mle-training 1"
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    run_all(exp_name)
