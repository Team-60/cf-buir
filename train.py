import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt

# buir dependencies
from buir.buir_id import BUIR_ID
from buir.utils import init_logger, get_logger, init_device, get_device
from buir.options import args_parser
from buir.dataset import NUM_ITEMS, NUM_USERS, get_test_train_interactions, init_data_matrices

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

# -------------- setup model --------------

model = BUIR_ID(NUM_USERS, NUM_ITEMS, args.latent_size, args.momentum)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logger.info("model initialized!")

# -------------- training --------------

epoch_tr_losses = []
for epoch in range(args.epochs):
    train_loss, train_samples = 0, 0
    model.train()
    for (b_users, b_items) in train_dataloader:
        b_users, b_items = b_users.to(device), b_items.to(device)
        optimizer.zero_grad()
        u_online, u_target, i_online, i_target = model((b_users, b_items))
        b_loss = model.get_loss((u_online, u_target, i_online, i_target))

        train_loss += b_loss.item() * b_users.shape[0]
        train_samples += b_users.shape[0]

        b_loss.backward()
        optimizer.step()
        model._update_target()
    train_loss /= train_samples
    epoch_tr_losses.append(train_loss)
    logger.info(f"train loss after epoch {epoch}: loss ({train_loss:.5f})")

    # !TODO: Add evaluation & early stopping

# -------------- save exp results --------------

plt.figure()
plt.plot(range(args.epochs), epoch_tr_losses, label="train losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training metrics")
plt.legend()
plt.savefig(f"{EXP_FOLDER}/training-plot.png")

torch.save(model, f"{EXP_FOLDER}/model.pt")
