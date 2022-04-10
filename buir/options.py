import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--epochs", type=int, default=50, help="rounds of training")
    args = parser.parse_args()
    return args
