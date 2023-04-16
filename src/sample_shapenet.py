import os
from distutils.dir_util import copy_tree
import random

_dataset_path = "/home/witold/Cargo/ShapeNetCore.v2"
_sample_percent = 0.1

_destination_path = f"/home/witold/Cargo/ShapeNetCore_{_sample_percent}"

if __name__ == "__main__":
    dir_to_copy = dict()
    for directory in os.listdir(_dataset_path):
        path = f"{_dataset_path}/{directory}"
        dir_to_copy[directory] = []
        if os.path.isdir(path):
            for instance in os.listdir(path):
                dir_to_copy[directory].append(instance)

    for k, v in dir_to_copy.items():
        reduced_files = random.choices(v, k=int(len(v) * _sample_percent))
        for r in reduced_files:
            copy_tree(
                    src=f"{_dataset_path}/{k}/{r}",
                    dst=f"{_destination_path}/{k}/{r}",
                )
