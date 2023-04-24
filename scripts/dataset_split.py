import glob
import random
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

"""
Tzn ja bym zrobiła % kategorii do traina, % kategorii do test, % do walidacji,
i potem z pozostałych % dzielony między train i test, % dzielony między train i walidacja 
i może jeszcze % dzielony między walidacja i test
"""

dataset_path = "../data/images/shapenet"


def get_partial_path_by_category(cat: str):
    depth = glob.glob(f"{dataset_path}/{cat}/depth/*.png")
    rgb = glob.glob(f"{dataset_path}/{cat}/rgb/*.png")
    depth = ['/'.join(path.split('/')[4:]) for path in depth]
    rgb = ['/'.join(path.split('/')[4:]) for path in rgb]
    return rgb, depth


def get_set(list_categories: Iterable) -> pd.DataFrame:
    rgb, depth = [], []
    for cat in list_categories:
        tmp_rgb, tmp_depth = get_partial_path_by_category(cat)
        rgb += tmp_rgb
        depth += tmp_depth
    return pd.DataFrame(
        data={"rgb_path": rgb, "depth_path": depth}
    )


def split_categories(list_categories: Iterable, set_1: pd.DataFrame, set_2: pd.DataFrame) -> Tuple:
    for cat in list_categories:
        tmp_set = get_set([cat])
        assert tmp_set.shape[0] % 30 == 0
        half = int(np.ceil((tmp_set.shape[0] / 30) / 2)) * 30
        return pd.concat([set_1, tmp_set.iloc[:half, :]]), pd.concat([set_2, tmp_set.iloc[half:, :]])


all_categories = set(os.listdir("../data/images/shapenet"))
all_categories.remove("camera.npy")
num_categories = len(all_categories)
random.seed(23)

# entire categories assigned to sets
_train_categories = 0.6
_test_categories = 0.15
_validation_categories = 0.1
# categories divided between particular sets
_train_test = 0.05
_train_eval = 0.05
_test_eval = 0.05

print(_train_categories + _test_categories + _validation_categories + _train_test + _train_eval + _test_eval)

# compute sizes
_train_size = int(round(_train_categories * num_categories, 0))
_test_size = int(round(_test_categories * num_categories, 0))
_validation_size = int(round(_validation_categories * num_categories, 0))
_train_test_size = int(round(_train_test * num_categories, 0))
_train_eval_size = int(round(_train_eval * num_categories, 0))
_test_eval_size = int(round(_test_eval * num_categories, 0))

summed_size = _train_size + _test_size + _validation_size + _train_test_size + _train_eval_size + _test_eval_size

# in case of bad rounding: add or remove categories from the general training set
if summed_size != num_categories:
    _train_size += (num_categories - summed_size)

assert _train_size + _test_size + _validation_size + _train_test_size + _train_eval_size + _test_eval_size == num_categories

train_categories = set(random.sample(all_categories, _train_size))
all_categories = all_categories - train_categories
test_categories = set(random.sample(all_categories, _test_size))
all_categories = all_categories - test_categories
validation_categories = set(random.sample(all_categories, _validation_size))
all_categories = all_categories - validation_categories
train_test_categories = set(random.sample(all_categories, _train_test_size))
all_categories = all_categories - train_test_categories
train_eval_categories = set(random.sample(all_categories, _train_eval_size))
all_categories = all_categories - train_eval_categories
test_eval_categories = set(random.sample(all_categories, _test_eval_size))
all_categories = all_categories - test_eval_categories

assert len(all_categories) == 0

train_set = get_set(train_categories)
test_set = get_set(test_categories)
eval_set = get_set(validation_categories)

train_set, test_set = split_categories(train_test_categories, train_set, test_set)
train_set, eval_set = split_categories(train_eval_categories, train_set, eval_set)
test_set, eval_set = split_categories(test_eval_categories, test_set, eval_set)

train_set.to_csv("../train_test_splits/train.csv", sep=';')
test_set.to_csv("../train_test_splits/test.csv", sep=';')
eval_set.to_csv("../train_test_splits/eval.csv", sep=';')
