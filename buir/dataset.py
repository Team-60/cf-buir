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
    - Ratings greater than mean + std
    """

    def get_mean(x: np.array) -> float:
        x_ = x[x > 0]
        return x_.sum() / len(x_)

    def get_std(x: np.array) -> float:
        x_ = x[x > 0]
        return x_.std()

    user_thresh = np.array([get_mean(data_mat[i]) + get_std(data_mat[i]) for i in range(NUM_USERS)])  # TODO: downcast?
    filtered_data_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    for i in range(NUM_USERS):
        for j in range(NUM_ITEMS):
            filtered_data_mat[i][j] = data_mat[i][j] if (data_mat[i][j] >= user_thresh[i]) else 0

    logger.info("filtered_data_mat initialized!")


def init_adj_matrix():
    """
    Initialize adj matrix
    """
    user_item_adj_mat = np.where(filtered_data_mat > 0, 1, 0)
    full_adj_mat = np.zeros((NUM_USERS + NUM_ITEMS, NUM_USERS + NUM_ITEMS))
    full_adj_mat[:NUM_USERS, NUM_USERS:] = user_item_adj_mat
    full_adj_mat[NUM_USERS:, :NUM_USERS] = user_item_adj_mat.T

    np.seterr(divide="ignore")
    D = np.power(np.sum(full_adj_mat, axis=1), -0.5)
    D[np.isinf(D)] = 0
    np.seterr(divide="warn")

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


def get_test_train_interactions(split_ratio: float) -> Tuple[InteractionDataset, InteractionDataset, List[Interaction], List[Interaction]]:
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

    train_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    test_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    for i, j in train_interactions:
        train_mat[i, j] = 1

    for i, j in test_interactions:
        test_mat[i, j] = 1

    logger.info(f"total train_interactions: {len(train_interactions)}, test_interactions: {len(test_interactions)}")
    return InteractionDataset(train_interactions), InteractionDataset(test_interactions), train_interactions, test_interactions


def get_adj_matrix() -> np.array:
    return init_adj_matrix()


# -------------- cold start stuff --------------


def get_test_train_interactions_cold_start(split_ratio: float) -> Tuple[InteractionDataset, InteractionDataset, List[Interaction], List[Interaction]]:
    """
    For each user, consider a user to be training with split_ratio probability
    """
    logger.debug(f"train test interactions requested, split_ratio: {split_ratio}")
    train_interactions, test_interactions = [], []
    choose_prob = np.random.choice(a=[True, False], size=NUM_USERS, p=[split_ratio, 1 - split_ratio])
    train_users, test_users = [], []
    for i in range(NUM_USERS):
        if choose_prob[i]:
            train_users.append(i)
        else:
            test_users.append(i)
        for j in range(NUM_ITEMS):
            if filtered_data_mat[i][j] == 0:
                continue
            if choose_prob[i]:
                train_interactions.append((i, j))
            else:
                test_interactions.append((i, j))

    train_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    test_mat = np.zeros((NUM_USERS, NUM_ITEMS))
    for i, j in train_interactions:
        train_mat[i, j] = 1

    for i, j in test_interactions:
        test_mat[i, j] = 1

    logger.info(f"total train_interactions: {len(train_interactions)}, test_interactions: {len(test_interactions)}")
    return InteractionDataset(train_interactions), InteractionDataset(test_interactions), train_interactions, test_interactions, np.array(train_users), np.array(test_users)


def get_zip_df() -> pd.DataFrame:
    """
    zip to lat, long
    """
    zip_data = codecs.open(f"data/zip_data.txt", "rU", "UTF-8")
    zip_df = pd.read_csv(zip_data)
    zip_df.set_index("ZIP", inplace=True)
    return zip_df


def form_metadata_df(zip_df: pd.DataFrame) -> np.array:
    """
    meta data for clustering
    """
    columns_ = ["user-id", "age", "gender", "occupation", "zip-code"]
    meta_data = codecs.open(f"data/ml-100k/u.user", "rU", "UTF-8")
    meta_df = pd.read_csv(meta_data, sep="|", encoding="utf-8", names=columns_)
    meta_df["user-id"] -= 1
    meta_df.set_index("user-id", inplace=True)
    meta_df.sort_index(inplace=True)
    # preprocessing
    lats, longs = [], []
    for uid in meta_df.index:
        try:
            zip_ = int(meta_df["zip-code"][uid])
            assert zip_ in zip_df.index
            lats.append(zip_df.loc[zip_]["LAT"])
            longs.append(zip_df.loc[zip_]["LNG"])
        except:
            lats.append(0)
            longs.append(0)

    meta_df.drop("zip-code", axis=1, inplace=True)
    meta_df["lats"] = lats
    meta_df["longs"] = longs

    one_hot_f = ["gender", "occupation"]
    for feat in one_hot_f:
        one_hot_ = pd.get_dummies(meta_df[feat], prefix=f"{feat}")
        meta_df.drop(feat, axis=1, inplace=True)
        meta_df = meta_df.join(one_hot_)
    meta_df = (meta_df - meta_df.min()) / (meta_df.max() - meta_df.min())
    meta_mat = meta_df.to_numpy()
    return meta_mat
