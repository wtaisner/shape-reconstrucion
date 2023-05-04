# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_path):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_generated_gt(generated_volume, gt_volume, save_path=None):
    generated_volume = generated_volume.squeeze().__ge__(0.5)
    gt_volume = gt_volume.squeeze().__ge__(0.5)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    ax1, ax2 = fig.add_subplot(gs[0, 0], projection='3d'), fig.add_subplot(gs[0, 1], projection='3d')
    ax1.set_aspect('equal')
    ax1.voxels(generated_volume, edgecolor="k")
    ax2.set_aspect('equal')
    ax2.voxels(gt_volume, edgecolor="k")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
