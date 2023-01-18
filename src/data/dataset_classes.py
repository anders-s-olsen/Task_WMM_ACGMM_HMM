import torch
import os
import numpy as np
from torch import nn, optim, backends
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from ..data.synthetic_generator import synthetic3D


class ALLdataset(Dataset):
    def __init__(self, hdf5_path, transform=None):

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label