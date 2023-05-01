import os.path
import pathlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
import multiprocessing as mp
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    """
    Voxelization part is based on:
    https://github.com/Yang7879/3D-RecGAN-extended/blob/master/Data_preprocess/depth_2_pc_2_vox.py
    """

    def __init__(
            self,
            datafile: str,
            vox_res: int = 64,
            data_path: str = "../data2/images/shapenet",
            voxel_path: str = "../data2/voxels/shapenet",

    ):
        self.data = pd.read_csv(datafile, sep=';', index_col=0)
        self.data_path = data_path
        self.vox_res = vox_res
        self.voxel_path = voxel_path

    def single_depth_2_pc(self, depth_path: str):
        depth = Image.open(os.path.join(self.data_path, depth_path)).convert("L")
        depth = np.asarray(depth, dtype=np.float32)
        depth /= 255.0
        # plt.imshow(depth, cmap='gray')
        # plt.show()

        h = depth.shape[0]
        w = depth.shape[1]

        fov = 49.124 / 2  # degree
        fx = w / (2.0 * np.tan(fov / 180.0 * np.pi))
        fy = h / (2.0 * np.tan(fov / 180.0 * np.pi))
        # k = np.array([[fx, 0, w / 2],
        #               [0, fy, h / 2],
        #               [0, 0, 1]], dtype=np.float32)

        xyz_pc = []
        for hi in range(h):
            for wi in range(w):
                if depth[hi, wi] > 5 or depth[hi, wi] == 0.0:
                    depth[hi, wi] = 0.0
                    continue
                x = -(wi - w / 2) * depth[hi, wi] / fx
                y = -(hi - h / 2) * depth[hi, wi] / fy
                z = depth[hi, wi]
                xyz_pc.append([x, y, z])

        xyz_pc = np.asarray(xyz_pc, dtype=np.float16)
        return xyz_pc

    def voxelization(self, pc_25d):
        self.vox_res = 64

        x_max = max(pc_25d[:, 0])
        x_min = min(pc_25d[:, 0])
        y_max = max(pc_25d[:, 1])
        y_min = min(pc_25d[:, 1])
        z_max = max(pc_25d[:, 2])
        z_min = min(pc_25d[:, 2])
        step = round(max([x_max - x_min, y_max - y_min, z_max - z_min]) / (self.vox_res - 1), 4)
        x_d_s = int((x_max - x_min) / step)
        y_d_s = int((y_max - y_min) / step)
        z_d_s = int((z_max - z_min) / step)

        vox = np.zeros((x_d_s + 1, y_d_s + 1, z_d_s + 1, 1), dtype=np.int8)
        for k, p in enumerate(pc_25d):
            xd = int((p[0] - x_min) / step)
            yd = int((p[1] - y_min) / step)
            zd = int((p[2] - z_min) / step)
            if xd >= self.vox_res or yd >= self.vox_res or zd >= self.vox_res:
                continue
            if xd > x_d_s or yd > y_d_s or zd > z_d_s:
                continue

            vox[xd, yd, zd, 0] = 1

        return vox

    @staticmethod
    def plot_from_voxels(voxels):
        if len(voxels.shape) > 3:
            x_d = voxels.shape[0]
            y_d = voxels.shape[1]
            z_d = voxels.shape[2]
            v = voxels[:, :, :, 0]
            v = np.reshape(v, (x_d, y_d, z_d))
        else:
            v = voxels
        x, y, z = v.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        plt.show()

    def voxel_grid_padding(self, a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        ori_vox_res = self.vox_res
        size = [ori_vox_res, ori_vox_res, ori_vox_res, channel]
        b = np.zeros(size, dtype=np.float32)

        bx_s = 0
        bx_e = size[0]
        by_s = 0
        by_e = size[1]
        bz_s = 0
        bz_e = size[2]
        ax_s = 0
        ax_e = x_d
        ay_s = 0
        ay_e = y_d
        az_s = 0
        az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e, :] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        return b

    def __len__(self):
        return len(self.data)

    def save_voxels(self, tmp_data):
        for idx, (rgb_path, depth_path) in tmp_data.iterrows():
            voxel_path = rgb_path.replace("/rgb/", "/voxel/")
            voxel_path = voxel_path.replace(".png", ".npz")
            point_cloud = self.single_depth_2_pc(depth_path)
            voxel_grid = self.voxelization(point_cloud)
            padded = torch.tensor(self.voxel_grid_padding(voxel_grid)).permute(3, 0, 1, 2).numpy()
            path = os.path.join(self.voxel_path, voxel_path)
            if not os.path.isdir("/".join(path.split("/")[:-1])):
                pathlib.Path("/".join(path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, padded)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data.iloc[idx, :]
        voxel_path = rgb_path.replace("/rgb/", "/voxel/")
        voxel_path = voxel_path.replace(".png", ".npz")
        voxel_grid = np.load(os.path.join(self.voxel_path, voxel_path))
        # self.plot_from_voxels(voxel_grid)
        return torch.tensor(voxel_grid)


if __name__ == "__main__":
    dataset = ShapeNetDataset("../train_test_splits/eval_001.csv")
    n = 10  # chunk row size
    list_df = [dataset.data[i:i + n] for i in range(0, dataset.data.shape[0], n)]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(dataset.save_voxels, list_df)
    # print(dataset.__getitem__(2137).shape)
