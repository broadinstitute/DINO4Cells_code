import os
import torch
import skimage.io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
t = torchvision.transforms.ToTensor()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


########################################################
## Re-arrange channels from tape format to stack tensor
########################################################

def fold_channels(image, channel_width, mode="drop"):
    # Expected input image shape: (h, w * c)
    # Output image shape: (h, w, c)
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")

    if mode == "ignore":
        # Keep all channels
        pass
    elif mode == "drop":
        # Drop mask channel (last)
        output = output[:, :, 0:-1]
    elif mode == "apply":
        # Use last channel as a binary mask
        mask = output["image"][:, :, -1:]
        output = output[:, :, 0:-1] * mask

    return t(output)


########################################################
## Dataset Class
########################################################

class SingleCellDataset(Dataset):
    """Single cell dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata.loc[idx, "Image_Name"])
        channel_width = self.metadata.iloc[idx, 'channel_width']
        image = skimage.io.imread(img_name)
        image = fold_channels(image, channel_width)

        label = self.metadata.loc[idx, "Target"]

        if self.transform:
            image = self.transform(image)

        return image, label

