from typing import Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob
import os
import multiprocessing as mp

from src.Pix2Vox.utils import binvox_rw
from src.Pix2Vox.utils.binvox_rw import Voxels

_dataset_path = "/home/witold/Cargo/ShapeNetTmp"  # datasets to take binvox files from
_target_dataset_path = "/home/witold/Cargo/ShapeNetVox32_sample_0.5_enhanced"  # dataset to save binvox files to


def down_sample(volume_path: Union[str, os.PathLike]) -> None:
    with open(os.path.join(_dataset_path, volume_path), 'rb') as f:
        volume = binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)

    steps = [0.25, 0.25, 0.25]  # original step sizes
    x, y, z = [steps[k] * np.arange(volume.shape[k]) for k in range(3)]  # original grid
    f = RegularGridInterpolator((x, y, z), volume)  # interpolator
    dx, dy, dz = 1.0, 1.0, 1.0  # new step sizes
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]  # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    new_values = f(new_grid)
    new_values = Voxels(new_values, new_values.shape, (0, 0, 0), 0, "xyz")
    save_path = os.path.join(_target_dataset_path, *volume_path.split("/")[:-2])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    binvox_rw.write(new_values, open(os.path.join(save_path, "model.binvox"), 'wb'))


if __name__ == "__main__":
    imgs_instances = ["/".join(x.split("/")[-4:]) for x in
                      glob.glob(f"{_dataset_path}/*/*/*/*.solid.binvox", recursive=True)]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        pool.map(down_sample, imgs_instances)
