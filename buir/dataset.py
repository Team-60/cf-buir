from typing import Tuple, List
from . import logger, Interaction
import codecs
import pandas as pd
import numpy as np

# -------------- form data matrix --------------

NUM_USERS = 943
NUM_ITEMS = 1682
COLUMNS = ["user-id", "item-id", "rating", "timestamp"]

doc_data = codecs.open(f"data/ml-100k/u.data", "rU", "UTF-8")
base_df = pd.read_csv(doc_data, sep="\t", encoding="utf-8", names=COLUMNS)
base_df["user-id"] -= 1
base_df["item-id"] -= 1

data_mat = np.zeros((NUM_USERS, NUM_ITEMS))
for i in range(len(base_df)):
    data_mat[base_df["user-id"][i]][base_df["item-id"][i]] = base_df["rating"][i]

logger.debug("data matrix initialized!")

# -------------- filter positive interactions for OCCF --------------
"""
Metric used for OCCF:
- Ratings greater than mean
"""


def get_mean(x: np.array) -> float:
    x_ = x[x > 0]
    return x_.sum() / len(x_)


user_thresh = np.array([get_mean(data_mat[i]) for i in range(NUM_USERS)])  # TODO: downcast?
filtered_data_mat = np.zeros((NUM_USERS, NUM_ITEMS))
for i in range(NUM_USERS):
    for j in range(NUM_ITEMS):
        filtered_data_mat[i][j] = data_mat[i][j] if (data_mat[i][j] >= user_thresh[i]) else 0

logger.debug("filtered_data_mat initialized!")

# -------------- form train test splits --------------


def get_test_train_interactions(split_ratio: float) -> Tuple[List[Interaction], List[Interaction]]:
    """
    For each user, consider a rating to be training with split_ratio probability
    """
    logger.debug(f"train test interactions requested, split_ratio: {split_ratio}")
    train_interactions, test_interactions = [], []
    choose_prob = np.random.choice(a=[True, False], size=(NUM_USERS, NUM_ITEMS), p=[split_ratio, 1 - split_ratio])
    for i in range(NUM_USERS):
        for j in range(NUM_ITEMS):
            if filtered_data_mat[i][j] == 0:
                continue
            if choose_prob[i][j]:
                train_interactions.append((i, j))
            else:
                test_interactions.append((i, j))

    logger.debug(f"Total train_interactions: {len(train_interactions)}, test_interactions: {len(test_interactions)}")
    return train_interactions, test_interactions