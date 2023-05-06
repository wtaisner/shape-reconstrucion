"""
Get an intersection of instances present in ShapeNet and downloaded part of shapenet containing binvox files.
"""
import glob
import os
from distutils.dir_util import copy_tree

_dataset_path = "/home/witold/Cargo/ShapeNetCore_0.1"
_binvox_path = "/home/witold/Cargo/ShapeNetVox32"

if __name__ == "__main__":
    imgs = ["/".join(x.split("/")[-2:]) for x in glob.glob(f"{_dataset_path}/*/*", recursive=True)]
    binvox = ["/".join(x.split("/")[-2:]) for x in glob.glob(f"{_binvox_path}/*/*", recursive=True)]

    common = set(imgs).intersection(set(binvox))
    for file in common:
        cat, instance = file.split("/")
        copy_tree(
            src=os.path.join(_dataset_path, cat, instance),
            dst=os.path.join(_binvox_path, cat, instance),
        )
