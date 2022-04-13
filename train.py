import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json

# buir dependencies
from buir.utils import init_logger, get_logger, init_device, get_device
from buir.options import args_parser
from buir.dataset import get_test_train_interactions, init_data_matrices

# -------------- set seed --------------

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# -------------- setup experiment --------------

args = args_parser()

EXP_FOLDER = f"experiments/{args.exp_name}"
os.makedirs(EXP_FOLDER)

init_logger(f"{EXP_FOLDER}/logs.log")
logger = get_logger()

with open(f"{EXP_FOLDER}/exp-info.json", "w") as fp:
    json.dump(vars(args), fp, indent=4)

init_device()
device = get_device()
logger.info(f"device used: {device}")

# -------------- get data --------------

init_data_matrices()
train_interactions_ds, test_interactions_ds = get_test_train_interactions(args.train_ratio)
train_dataloader = DataLoader(
    train_interactions_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_interactions_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)
