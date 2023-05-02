import torch.utils.data as data
import torchvision
import torch
import os
import numpy as np
import os.path
import pandas as pd
from skimage import io
from pathlib import Path
from functools import partial
from sklearn.preprocessing import StandardScaler
from utils import cellpainting_dataset


t = torchvision.transforms.ToTensor()


def tensor_loader(path, training=True):
    return torch.load(path)


def default_loader(path, training=True):
    return t(io.imread(path))


def one_channel_loader(path, training=True):
    img = io.imread(path)
    if training:
        ch = np.random.randint(0, 4)
        return t(img[:, :, ch])
    else:
        return [t(img[:, :, i]) for i in range(img.shape[-1])]


def two_channel_loader(path, training=True):
    img = io.imread(path)
    if training:
        out = img[:, :, 0:2]
        ch = np.random.randint(0, 4)
        out[:, :, 0] = t(img[:, :, ch])
    else:
        out = [t(img[:, :, [i, 1]]) for i in range(4)]
    return out


def protein_channel_loader(path, training=True):
    img = io.imread(path)
    return t(img[:, :, 1])


def single_channel_loader(
    path,
    training=True,
    channel=0,
    triple=True,
):
    img = io.imread(path)
    # img = io.imread(path).transpose(1, 2, 0)
    img = img[:, :, [channel]]
    if triple:
        img = np.repeat(img, 3, axis=2)
    return t(img.astype(np.uint8))


# def single_channel_loader(
#     path,
#     training=True,
#     channel=0,
# ):
#     img = io.imread(path)
#     # img = io.imread(path).transpose(1, 2, 0)
#     img = img[:, :, [channel]]
#     img = np.repeat(img, 3, axis=2)
#     return img.astype(np.uint8)


def norm_loader(path, training=True):
    img = io.imread(path)
    img = img.astype(float)
    img -= np.min(img, axis=(0, 1))
    img *= 255 / np.max(img, axis=(0, 1))
    return img.astype(np.uint8)


def pandas_reader_binary_labels(flist, target_labels=None, sample_single_cells=False):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    if isinstance(flist, pd.DataFrame):
        files = flist
    else:
        files = pd.read_csv(flist)[["file", "ID", "cell_type"] + target_labels]
    target_matrix = files[target_labels].values.astype(int)
    file_names = files["file"].values
    IDs = files["ID"].values
    cell_lines = files["cell_type"].values
    if sample_single_cells:
        ID_groups = files.groupby("ID").groups
        IDs = sorted(ID_groups.keys())
        cell_lines = [cell_lines[ID_groups[ID]] for ID in sorted(ID_groups.keys())]
        file_names = [file_names[ID_groups[ID]] for ID in sorted(ID_groups.keys())]
        target_matrix = [
            target_matrix[ID_groups[ID]] for ID in sorted(ID_groups.keys())
        ]
    imlist = []
    for impath, ID, cell_line, target in zip(
        file_names, IDs, cell_lines, target_matrix
    ):
        imlist.append((impath, target, cell_line, ID))
    return imlist


def pandas_reader(flist, ids=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    if isinstance(flist, pd.DataFrame):
        files = flist
    else:
        files = pd.read_csv(flist)[["file", "protein_location", "cell_type", "ID"]]
    if type(ids) is not type(None):
        files = files[files.ID.isin(ids)]
    files = np.array(files.to_records(index=False))
    imlist = []
    for impath, protein_location, cell_type, ID in files:
        imlist.append((impath, protein_location, cell_type, ID))
    return imlist


def pandas_reader_no_labels(flist, target_labels=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    files = pd.read_csv(flist)[["file", "ID"]]
    print(len(files))
    files = np.array(files.to_records(index=False))
    imlist = []
    for impath, imlabel in files:
        imlist.append((impath, imlabel))
    return imlist


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(",")
            imlist.append((impath, imlabel))
    return imlist


def pandas_reader_only_file(flist, ids=None, sample_single_cells=False):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    if isinstance(flist, pd.DataFrame):
        files = flist
    else:
        files = pd.read_csv(flist)[["file", "ID"]]
    if type(ids) is not type(None):
        files = files[files.ID.isin(ids)]
    if sample_single_cells:
        ID_groups = files.groupby("ID").groups
        IDs = sorted(ID_groups.keys())
        file_names = [
            files.iloc[ID_groups[ID]]["file"].values for ID in sorted(ID_groups.keys())
        ]
        imlist = []
        for impath, ID in zip(file_names, IDs):
            imlist.append((impath, ID))
    else:
        files = np.array(files.to_records(index=False))
        imlist = []
        for impath, ID in files:
            imlist.append((impath, ID))
    return imlist


class ImageFileList(data.Dataset):
    def __init__(
        self,
        flist,
        root,
        transform=None,
        balance=True,
        flist_reader=pandas_reader_binary_labels,
        loader=default_loader,
        training=True,
        with_labels=True,
        target_labels=None,
        single_cells=False,
        sample_single_cells=False,
    ):
        self.balance = balance
        self.imdf = pd.read_csv(flist)
        self.idx = []
        if type(target_labels) is not type(None):
            self.target_labels = sorted(target_labels)
            self.with_target_labels = True
            self.imlist = flist_reader(flist, self.target_labels)
        else:
            self.imlist = flist_reader(flist)
            self.with_target_labels = False
            self.target_labels = None
        if balance:
            self.parse_labels()
        self.transform = transform
        self.loader = loader
        self.training = training
        self.with_labels = with_labels
        self.root = root
        self.single_cells = single_cells
        self.sample_single_cells = sample_single_cells

    def parse_labels(self):
        if type(self.target_labels) is not type(None):
            self.unique = self.target_labels
        else:
            labels = [x for x in self.imdf.protein_location.apply(eval)]
            self.unique = set()
            result = [[self.unique.add(x) for x in y] for y in labels]
            self.unique = list(self.unique)
            self.unique.sort()
            for u in self.unique:
                self.imdf[u] = False
            for k in range(len(labels)):
                for c in labels[k]:
                    self.imdf.loc[k, c] = True
        stats = pd.DataFrame(
            data=[{"class": u, "freq": np.sum(self.imdf[u])} for u in self.unique]
        )
        self.stats = stats.sort_values(by="freq")
        N = int(self.stats.freq.mean())
        print("Sampling", N, "images per class for", len(self.unique), "classes")
        self.N = N * len(self.unique)

    def __getitem__(self, index):
        # Mapping index to a virtual table of classes
        if self.balance:
            class_id = self.unique[index % len(self.unique)]
            sample_idx = self.imdf[self.imdf[class_id]].sample(n=1).index[0]
        else:
            sample_idx = index

        # Identify the sample
        if type(self.target_labels) is not type(None):
            impath, protein, cell, ID = self.imlist[sample_idx]
        elif self.with_labels:
            impath, protein, cell, ID = self.imlist[sample_idx]
        else:
            if self.sample_single_cells:
                files, ID = self.imlist[sample_idx]
                # Randomly choose a cell from the whole image
                impath = np.random.choice(files, 1)[0]
            else:
                impath, ID = self.imlist[sample_idx]
        img = self.loader(self.root + impath, self.training)

        # Transform the image
        if self.transform is not None:
            if isinstance(img, list):
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        # Return the item
        if self.training:
            return img, ID
        elif type(self.target_labels) is not type(None):
            return img, protein.astype(int), cell, ID
        elif self.with_labels:
            return img, protein, cell, ID
        elif self.single_cells:
            return img, ID, Path(impath).stem
        else:
            return img, ID

    def __len__(self):
        return len(self.imlist)


class AutoBalancedPrecomputedFeatures(data.Dataset):
    def __init__(self, source, balance, target_column, scaler=None, **kwargs):
        features, proteins, cells, IDs = torch.load(source)

        if isinstance(features, np.ndarray):
            self.features = torch.Tensor(features)
        elif isinstance(features, torch.Tensor):
            self.features = features.detach().cpu()
        self.IDs = np.array(IDs)
        self.proteins = proteins
        self.cells = cells
        if target_column == "proteins":
            if isinstance(proteins, np.ndarray):
                self.target = torch.Tensor(proteins)
            elif isinstance(proteins, torch.Tensor):
                self.target = proteins.detach().cpu()
        elif target_column == "cells":
            if isinstance(cells, np.ndarray):
                self.target = torch.Tensor(self.cells)
            elif isinstance(cells, torch.Tensor):
                self.target = self.cells.detach().cpu()
            # the following line removes all IDs from the origianl kaggle
            # competition (Since they had no cell type)
            indices = np.where(pd.DataFrame(IDs)[0].str.contains("-") == False)[0]
            self.features = self.features[indices, :]
            self.proteins = self.proteins[indices, :]
            self.cells = self.cells[indices, :]
            self.IDs = self.IDs[indices]
            self.target = self.target[indices, :]
        self.idx = []
        self.df = pd.DataFrame(range(len(self.features)), columns=["ind"])
        if balance:
            self.parse_labels()
        self.balance = balance

    def scale_features(self, scaler):
        self.scaler = scaler
        if self.scaler == "find_statistics":
            print("scaling training data!")
            self.scaler = StandardScaler().fit(self.features)
            self.features = torch.Tensor(self.scaler.transform(self.features.numpy()))
        elif self.scaler is not None:
            print("scaling validation / testing data!")
            self.features = torch.Tensor(self.scaler.transform(self.features.numpy()))

    def parse_labels(self):
        stats = pd.DataFrame(
            data=[
                {"class": u, "freq": self.target[:, u].sum().item()}
                for u in range(self.target.shape[1])
            ]
        )
        self.stats = stats.sort_values(by="freq")
        N = int(self.stats.freq.mean())
        print("Sampling", N, "samples per class for", self.target.shape[1], "classes")
        self.N = N * self.target.shape[1]

    def __getitem__(self, index):
        # Mapping index to a virtual table of classes
        # class_id = list(range(self.target.shape[1]))[index % self.target.shape[1]]
        class_id = np.random.choice(list(range(self.target.shape[1])))
        while self.target[:, class_id].sum() == 0:
            class_id = np.random.choice(list(range(self.target.shape[1])))
        if self.balance:
            sample_idx = np.random.choice(np.where(self.target[:, class_id])[0], 1)
        else:
            sample_idx = index
        # sample_idx = self.df[self.df[str(class_id)]].sample(n=1).index[0]
        return (
            self.features[sample_idx],
            self.target[sample_idx],
        )

    def __len__(self):
        return len(self.df)


class AutoBalancedFileList(ImageFileList):
    def __init__(
        self,
        flist,
        root,
        transform=None,
        flist_reader=None,
        loader=default_loader,
        training=True,
        with_labels=True,
        target_labels=None,
    ):
        self.target_labels = sorted(target_labels)
        self.imdf = pd.read_csv(flist)
        self.parse_labels()
        self.transform = transform
        self.loader = loader
        self.training = training
        self.with_labels = with_labels
        self.root = root
        self.idx = []

    def parse_labels(self):
        if type(self.target_labels) is not type(None):
            self.unique = self.target_labels
        else:
            labels = [x for x in self.imdf.protein_location.apply(eval)]
            self.unique = set()
            result = [[self.unique.add(x) for x in y] for y in labels]
            self.unique = list(self.unique)
            self.unique.sort()
            for u in self.unique:
                self.imdf[u] = False
            for k in range(len(labels)):
                for c in labels[k]:
                    self.imdf.loc[k, c] = True
        stats = pd.DataFrame(
            data=[{"class": u, "freq": np.sum(self.imdf[u])} for u in self.unique]
        )
        self.stats = stats.sort_values(by="freq")
        N = int(self.stats.freq.mean())
        print("Sampling", N, "images per class for", len(self.unique), "classes")
        self.N = N * len(self.unique)

    def __getitem__(self, index):
        # Mapping index to a virtual table of classes
        class_id = self.unique[index % len(self.unique)]
        sample_idx = self.imdf[self.imdf[class_id]].sample(n=1).index[0]

        # Identify the sample
        if type(self.target_labels) is not type(None):
            impath, ID = self.imdf.iloc[sample_idx][["file", "ID"]]
            protein = self.imdf.iloc[sample_idx][self.target_labels].values.astype(int)
        elif self.with_labels:
            impath, protein, cell, ID = self.imdf.iloc[sample_idx][
                ["file", "protein_location", "cell_type", "ID"]
            ]
        else:
            impath, ID = self.imdf.iloc[sample_idx][["file", "ID"]]
        img = self.loader(self.root + impath, self.training)

        # Transform the image
        if self.transform is not None:
            if isinstance(img, list):
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        # Return the item
        if self.training:
            return img, protein.astype(int)
        elif type(self.target_labels) is not type(None):
            return img, protein.astype(int), cell, ID
        elif self.with_labels:
            return img, protein, cell, ID
        else:
            return img, ID

    def __len__(self):
        return self.N


data_loaders = {
    "HPA": ImageFileList,
    "HPABalanced": AutoBalancedFileList,
    "CellPainting": cellpainting_dataset.SingleCellDataset,
}

image_modes = {
    "normalized_3_channels": default_loader,
    "normalized_4_channels": default_loader,
    "unnormalized_4_channels": norm_loader,
    "single_channel_r": partial(single_channel_loader, channel=0, triple=True),
    "single_channel_g": partial(single_channel_loader, channel=1, triple=True),
    "single_channel_b": partial(single_channel_loader, channel=2, triple=True),
    "single_channel_y": partial(single_channel_loader, channel=3, triple=True),
    "single_channel_r_no_triple": partial(
        single_channel_loader, channel=0, triple=False
    ),
    "single_channel_g_no_triple": partial(
        single_channel_loader, channel=1, triple=False
    ),
    "single_channel_b_no_triple": partial(
        single_channel_loader, channel=2, triple=False
    ),
    "single_channel_y_no_triple": partial(
        single_channel_loader, channel=3, triple=False
    ),
    "one_channel": one_channel_loader,
    "two_channels": two_channel_loader,
    "protein_channel": protein_channel_loader,
}

readers = {True: pandas_reader_binary_labels, False: pandas_reader_no_labels}
# readers = {True: pandas_reader, False: pandas_reader_no_labels}


def scKaggle_loader(fn, reader=io.imread):
    image = reader(fn)
    return image


def scKaggle_df_reader(csvpath):
    """
    Used for training MLP for scKaggle competition.
    Will hijack ID in ImageFileList as labels...
    """
    df = pd.read_csv(csvpath)
    targs = torch.tensor(df.iloc[:, 9:].values.astype(np.float32))
    lbls = df["Path"].values
    imlist = [[impath, targ] for impath, targ in zip(lbls, targs)]
    return imlist
