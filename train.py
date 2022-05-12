import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# buir dependencies
from models.BUIR_ID import BUIR_ID
from models.BUIR_NB import BUIR_NB
from buir.utils import init_logger, get_logger, init_device, get_device
from buir.options import args_parser
from buir.dataset import NUM_ITEMS, NUM_USERS, form_metadata_df, get_test_train_interactions, get_test_train_interactions_cold_start, get_zip_df, init_data_matrices, get_adj_matrix
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
if args.cold_start:
    train_interactions_ds, test_interactions_ds, train_mat, test_mat, train_users, test_users = get_test_train_interactions_cold_start(args.train_ratio)
else:
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

# -------------- setup model --------------

if args.model == "buir-id":
    model = BUIR_ID(NUM_USERS, NUM_ITEMS, args.latent_size, args.momentum)
elif args.model == "buir-nb":
    norm_adj_mat = get_adj_matrix()
    model = BUIR_NB(NUM_USERS, NUM_ITEMS, args.latent_size, norm_adj_mat, args.momentum)
else:
    logger.error("Invalid model type: {} -- chocies : 'buir-nb', 'buir-id' (default)".format(args.model))
    exit(1)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logger.info("model initialized!")

# -------------- training --------------

epoch_tr_losses = []
epoch_te_losses = []
eval_results = []
for epoch in range(args.epochs):
    logger.info("======================")
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

# -------------- cold start -------------

logger.info("cold start problem init!")

# perform clustering
zip_df = get_zip_df()
meta_mat = form_metadata_df(zip_df)
train_meta_mat, test_meta_mat = meta_mat[train_users], meta_mat[test_users]
kmeans = KMeans(n_clusters=args.cold_start_clusters, random_state=0).fit(train_meta_mat)
neighbours = kmeans.predict(test_meta_mat)

logger.info("kmeans performed for cold start")

# metrics
metric_names = ["P"]
metric_vals = [10, 20, 50]
metrics = {}
for mn in metric_names:
    for mv in metric_vals:
        metrics[f"{mn}{mv}"] = []

# evaluate
with torch.no_grad():
    for _n_uidx in range(len(test_users)):
        test_uidx = test_users[_n_uidx]
        test_uratings = test_mat[test_uidx]

        # calculate mean embedding
        neighbouring_users = train_users[kmeans.labels_ == neighbours[_n_uidx]]
        u_online = 0
        for _neigh_user in neighbouring_users:
            u_online += model.uo_encoder(torch.tensor([_neigh_user])).squeeze()
        u_online /= len(neighbouring_users)

        # calculate scores
        scores = []
        for _item_idx in range(len(test_uratings)):
            if test_uratings[_item_idx] == 0:
                scores.append(-np.inf)
                continue
            i_online = model.io_encoder(torch.tensor([_item_idx])).squeeze()
            u_online_p = model.predictor(u_online)
            i_online_p = model.predictor(i_online)
            score_ = torch.sum(u_online_p * i_online) + torch.sum(i_online_p * u_online)
            scores.append(score_.item())

        # calculate metrics
        # top-k matches (P)
        scores_sorted = np.argsort(scores)[::-1]
        test_ratings_sorted = np.argsort(test_uratings)[::-1]
        for mv in metric_vals:
            metrics[f"P{mv}"].append(np.isin(scores_sorted[:mv], test_ratings_sorted[:mv]).sum() / mv)

for mn in metric_names:
    for mv in metric_vals:
        metrics[f"{mn}{mv}"] = np.mean(metrics[f"{mn}{mv}"])

logger.info(f"COLD START METRICS: {metrics}")
