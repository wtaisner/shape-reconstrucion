"""
A utility script used to split the dataset into training, test and evaluation sets.
"""
import glob
import random
import os
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd

from src.utils import read_config

random.seed(23)

cfg = read_config("../config/dataset_split.yaml")
dataset_path = cfg["dataset_path"]
num_views = cfg["num_views"]


# _validation_categories = cfg["validation_categories"]
# categories divided between particular sets
# _train_test = cfg["train_test"]
# _train_eval = cfg["train_eval"]
# _test_eval = cfg["test_eval"]


def get_partial_path_by_category(cat: str) -> Tuple[List, List]:
    """
    Returns a tuple of lists containing paths to RGB and depth images for a given category.
    :param cat: name of the category
    :return: two lists containing paths to RGB and depth images
    """
    depth = glob.glob(f"{dataset_path}/{cat}/depth/*.png")
    rgb = glob.glob(f"{dataset_path}/{cat}/rgb/*.png")
    depth = ['/'.join(path.split('/')[4:]) for path in depth]
    rgb = ['/'.join(path.split('/')[4:]) for path in rgb]
    return rgb, depth


def get_set(list_categories: Iterable) -> pd.DataFrame:
    """
    Returns a dataframe containing paths to RGB and depth images for a given list of categories.
    :param list_categories: any iterable containing names of categories
    :return: pandas dataframe containing paths to RGB and depth images
    """
    rgb, depth = [], []
    for cat in list_categories:
        tmp_rgb, tmp_depth = get_partial_path_by_category(cat)
        rgb += tmp_rgb
        depth += tmp_depth
    return pd.DataFrame(
        data={"rgb_path": rgb, "depth_path": depth}
    )


def split_categories(
        list_categories: Iterable,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        eval_set: pd.DataFrame,
        num_views: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a given list of categories between two sets.
    :param list_categories: list of categories to be split
    :param set_1: set to which the first half of categories will be added
    :param set_2: set to which the second half of categories will be added
    :param num_views: number of views per object (to perform a good division
    :return: two dataframes appended with split categories
    """
    for cat in list_categories:
        tmp_set = get_set([cat])
        assert tmp_set.shape[0] % num_views == 0
        num_instances = tmp_set.shape[0] / num_views
        train_instances, test_instances, eval_instances = np.ceil(num_instances * cfg["category_train_percent"]), \
            np.ceil(num_instances * cfg["category_test_percent"]), \
            np.ceil(num_instances * cfg["category_eval_percent"])
        if train_instances + test_instances + eval_instances > num_instances:
            test_instances -= -(num_instances - test_instances - train_instances - eval_instances)
        assert train_instances + test_instances + eval_instances == num_instances
        train_part = tmp_set.iloc[:int(train_instances * num_views)]
        test_part = tmp_set.iloc[int(train_instances * num_views):int((train_instances + test_instances) * num_views)]
        eval_part = tmp_set.iloc[int((train_instances + test_instances) * num_views):]

        return pd.concat([train_set, train_part]), pd.concat([test_set, test_part]), pd.concat([eval_set, eval_part])


if __name__ == "__main__":
    _train_categories = cfg["train_categories"]
    _test_categories = cfg["test_categories"]
    all_categories = set(os.listdir(dataset_path))
    all_categories.remove("camera.npy")
    num_categories = len(all_categories)

    # compute sizes
    _train_size = int(round(_train_categories * num_categories, 0))
    _test_size = int(round(_test_categories * num_categories, 0))

    summed_size = _train_size + _test_size
    # in case of bad rounding: add or remove categories from the general training set
    if summed_size != num_categories:
        _train_size += (num_categories - summed_size)

    assert _train_size + _test_size == num_categories

    train_categories = set(random.sample(all_categories, _train_size))
    all_categories = all_categories - train_categories
    test_categories = set(random.sample(all_categories, _test_size))
    all_categories = all_categories - test_categories

    assert len(all_categories) == 0

    train_set = get_set(train_categories)
    test_set = get_set(test_categories)
    eval_set = pd.DataFrame(columns=["rgb_path", "depth_path"])

    train_set, test_set, eval_set = split_categories(train_categories, train_set, test_set, eval_set, num_views)
    # train_set, eval_set = split_categories(train_eval_categories, train_set, eval_set, num_views)
    # test_set, eval_set = split_categories(test_eval_categories, test_set, eval_set, num_views)

    train_set.to_csv("../train_test_splits/train_vox32_10_new_v2_final.csv", sep=';')
    test_set.to_csv("../train_test_splits/test_vox32_10_new_v2_final.csv", sep=';')
    eval_set.to_csv("../train_test_splits/eval_vox32_10_new_v2_final.csv", sep=';')
