from typing import Tuple, List
import codecs
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from . import Interaction
from .utils import get_logger

NUM_USERS = 943
NUM_ITEMS = 1682
data_mat: np.array
filtered_data_mat: np.array

logger = get_logger()


def init_data_matrices():
    """
    Reads and initializes data matrices
    """
    global data_mat
    global filtered_data_mat

    # -------------- form data matrix --------------

    COLUMNS = ["user-id", "item-id", "rating", "timestamp"]

    doc_data = codecs.open(f"data/ml-100k/u.data", "rU", "UTF-8")
    base_df = pd.read_csv(doc_data, sep="\t", encoding="utf-8", names=COLUMNS)
    base_df["user-id"] -= 1
    base_df["item-id"] -= 1

    data_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    for i in range(len(base_df)):
        data_mat[base_df["user-id"][i]][base_df["item-id"][i]] = base_df["rating"][i]

    logger.info("data matrix initialized!")

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

    logger.info("filtered_data_mat initialized!")

def init_adj_matrix():
    
    user_item_adj_mat = np.where(filtered_data_mat > 0, 1, 0)
    full_adj_mat = np.zeros((NUM_USERS + NUM_ITEMS, NUM_USERS + NUM_ITEMS))
    full_adj_mat[:NUM_USERS, NUM_USERS:] = user_item_adj_mat
    full_adj_mat[NUM_USERS:, :NUM_USERS] = user_item_adj_mat.T

    np.seterr(divide='ignore')
    D = np.power(np.sum(full_adj_mat, axis=1), -0.5)
    D[np.isinf(D)] = 0
    np.seterr(divide='warn')

    D_mat = np.diag(D)
    norm_adj_mat = D_mat @ full_adj_mat @ D_mat

    # TODO: self loop
    self_loop = False
    if self_loop:
        norm_adj_mat += np.diag(np.ones(NUM_USERS + NUM_ITEMS))

    return norm_adj_mat

# -------------- form train test splits --------------


class InteractionDataset(Dataset):
    """
    Converts a list of interactions into dataset
    """

    def __init__(self, interactions: List[Interaction]):
        super(InteractionDataset, self).__init__()
        self.interactions = interactions

    def __getitem__(self, index):
        _user, _item = self.interactions[index]
        return torch.tensor(_user), torch.tensor(_item)

    def __len__(self):
        return len(self.interactions)


def get_test_train_interactions(split_ratio: float) -> Tuple[InteractionDataset, InteractionDataset]:
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

    logger.info(f"total train_interactions: {len(train_interactions)}, test_interactions: {len(test_interactions)}")
    return InteractionDataset(train_interactions), InteractionDataset(test_interactions)

def get_adj_matrix() -> np.array:
    return init_adj_matrix()
