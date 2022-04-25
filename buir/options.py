import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default", help="experiment name")
    parser.add_argument("--exp_disc", type=str, default="default", help="description")

    parser.add_argument("--model", type=str, default="buir-id", help="type of model used")
    parser.add_argument("--latent_size", type=int, default=250, help="latent embeddings size")
    parser.add_argument("--epochs", type=int, default=50, help="rounds of training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--momentum", type=float, default=0.995, help="target encoder momentum")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio")

    parser.add_argument("--num_workers", type=int, default=0, help="num workers dataloader")
    args = parser.parse_args()
    return args
