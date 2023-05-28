"""
A utility script to sample a subset of the ShapeNet dataset.
"""
import os
import random

from distutils.dir_util import copy_tree

from src.utils import read_config

cfg = read_config("../config/sample_shapenet.yaml")


_destination_path = f"{cfg['destination_path']}_{cfg['sample_percent']}"

if __name__ == "__main__":
    dir_to_copy = dict()
    for directory in os.listdir(cfg['dataset_path']):
        path = f"{cfg['dataset_path']}/{directory}"
        dir_to_copy[directory] = []
        if os.path.isdir(path):
            for instance in os.listdir(path):
                dir_to_copy[directory].append(instance)

    for k, v in dir_to_copy.items():
        reduced_files = random.choices(v, k=int(len(v) * cfg["sample_percent"]))
        for r in reduced_files:
            copy_tree(
                    src=f"{cfg['dataset_path']}/{k}/{r}",
                    dst=f"{_destination_path}/{k}/{r}",
                )
