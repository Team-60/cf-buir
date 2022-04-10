import numpy as np
import torch
# buir dependencies
from buir.options import args_parser
from buir.dataset import get_test_train_interactions

# -------------- set seed --------------

np.random.seed(0)
torch.manual_seed(0)

# -------------- get args --------------

args = args_parser()

# -------------- get data --------------

train_interactions, test_interactions = get_test_train_interactions(args.train_ratio)