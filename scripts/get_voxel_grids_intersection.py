"""
Get an intersection of instances present in ShapeNet and downloaded part of shapenet containing binvox files.
"""
import glob
import os
from distutils.dir_util import copy_tree

from src.utils import read_config

if __name__ == "__main__":
    
    cfg = read_config("../config/voxel_grid_intersection.yaml")
    
    imgs = ["/".join(x.split("/")[-2:]) for x in glob.glob(f"{cfg['dataset_path']}/*/*", recursive=True)]
    binvox = ["/".join(x.split("/")[-2:]) for x in glob.glob(f"{cfg['binvox_path']}/*/*", recursive=True)]

    common = set(imgs).intersection(set(binvox))
    for file in common:
        cat, instance = file.split("/")
        copy_tree(
            src=os.path.join(cfg['dataset_path'], cat, instance),
            dst=os.path.join(cfg['target_path'], cat, instance),
        )
