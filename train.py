import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt

# buir dependencies
from models.BUIR_ID import BUIR_ID
from models.BUIR_NB import BUIR_NB
from buir.utils import init_logger, get_logger, init_device, get_device
from buir.options import args_parser
from buir.dataset import NUM_ITEMS, NUM_USERS, get_test_train_interactions, init_data_matrices, get_adj_matrix
from buir.evaluation import evaluate, print_eval_results, plot_eval_results

# -------------- set seed --------------

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# -------------- setup experiment --------------

args = args_parser()

EXP_FOLDER = f"experiments/{args.exp_name}"
os.makedirs(EXP_FOLDER, exist_ok=True)

init_logger(f"{EXP_FOLDER}/logs.log")
logger = get_logger()

with open(f"{EXP_FOLDER}/exp-info.json", "w") as fp:
    json.dump(vars(args), fp, indent=4)

init_device()
device = get_device()
logger.info(f"device used: {device}")

# -------------- get data --------------

init_data_matrices()
train_interactions_ds, test_interactions_ds, train_mat, test_mat = get_test_train_interactions(args.train_ratio)
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
norm_adj_mat = get_adj_matrix()

# -------------- setup model --------------

if args.model == 'buir-id':
    model = BUIR_ID(NUM_USERS, NUM_ITEMS, args.latent_size, args.momentum)
elif args.model == 'buir-nb':
    model = BUIR_NB(NUM_USERS, NUM_ITEMS, args.latent_size, norm_adj_mat, args.momentum)
else:
    logger.info("Invalid model type: {} -- chocies : 'buir-nb', 'buir-id' (default)".format(args.model))
    exit(1)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logger.info("model initialized!")

# -------------- training --------------

epoch_tr_losses = []
epoch_te_losses = []
eval_results = []
for epoch in range(args.epochs):
    logger.info('======================')
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

    model.eval()
    test_loss, test_samples = 0, 0
    with torch.no_grad():
        for (b_users, b_items) in test_dataloader:
            b_users, b_items = b_users.to(device), b_items.to(device)
            u_online, u_target, i_online, i_target = model((b_users, b_items))
            b_loss = model.get_loss((u_online, u_target, i_online, i_target))

            test_loss += b_loss.item() * b_users.shape[0]
            test_samples += b_users.shape[0]
        
        eval_result = evaluate(model, test_dataloader, train_mat, None, test_mat)
        print_eval_results(logger, eval_result)

        test_loss /= test_samples
        epoch_te_losses.append(test_loss)
        logger.info(f"test loss after epoch {epoch}: loss ({test_loss:.5f})")
        eval_results.append(eval_result)

    # !TODO: Add evaluation & early stopping

# -------------- save exp results --------------

plt.figure()
plt.plot(range(args.epochs), epoch_tr_losses, label="train losses")
plt.plot(range(args.epochs), epoch_te_losses, label="test losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training metrics")
plt.legend()
plt.savefig(f"{EXP_FOLDER}/loss-plot.png")

plot_eval_results(plt, EXP_FOLDER, eval_results)

# -------------- saving model -------------
torch.save(model, f"{EXP_FOLDER}/model.pt")


