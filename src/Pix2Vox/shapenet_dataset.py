"""
Implementation of the ShapeNet dataset class for the Pix2Vox model.
"""
import os
from typing import Union

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.Pix2Vox.utils import binvox_rw


class ShapeNetDataset(Dataset):
    def __init__(self, data_file: Union[str, pd.DataFrame], img_path: str, models_path: str, transforms=None):
        """
        Constructor for the ShapeNet dataset class.
        :param data_file: path to the csv file containing the data or the dataframe itself
        :param img_path: path to the images
        :param models_path: path to the models
        :param transforms: transformations to apply to the images
        """
        if type(data_file) is str:
            data = pd.read_csv(data_file, sep=';', index_col=0)
            self.data = list(data['depth_path'])
        else:
            self.data = data_file
        self.img_path = img_path
        self.models_path = models_path
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        depth_path = os.path.join(self.img_path, self.data[idx])
        taxonomy_name, taxonomy_sample = self.data[idx].split('/')[0], self.data[idx].split('/')[-1]
        sample_name = taxonomy_sample.split('_')[0]
        volume_path = os.path.join(self.models_path, taxonomy_name, sample_name, 'model.binvox')
        img = [cv2.resize(cv2.imread(depth_path).astype(np.float32), (224, 224)) / 255.]
        if self.transforms:
            img = self.transforms(img)
        with open(volume_path, 'rb') as f:
            volume = binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)
        return taxonomy_name, sample_name, np.asarray(img), volume
